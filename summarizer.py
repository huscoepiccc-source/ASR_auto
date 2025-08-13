# -*- coding: utf-8 -*-
"""
summarizer.py
- Robust local summarization for long markdown transcripts on Windows + CUDA.
- Backends: 'transformers' (default) and 'llama_cpp' (GGUF).
- Safe fallbacks: 4bit (bitsandbytes) -> fp16 on GPU -> CPU.
- Map-Reduce style chunked summarization, Qwen-friendly prompts.
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import os
import sys
import math
import traceback

# Optional imports deferred inside loader functions to avoid hard fails.


# -------------------------------
# Configuration dataclasses
# -------------------------------

@dataclass
class GenConfig:
    max_new_tokens: int = 800
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = False
    repetition_penalty: float = 1.05


@dataclass
class ChunkConfig:
    # Token-level chunking to avoid OOM
    chunk_tokens: int = 1800
    overlap_tokens: int = 150
    reduce_tokens: int = 1600  # "reduce" phase input size


@dataclass
class LoadConfig:
    # device: 'auto' | 'cuda' | 'cpu' | 'mps' (mac)
    device: str = "auto"
    prefer_4bit: bool = True  # try 4bit first if possible
    trust_remote_code: bool = True


# -------------------------------
# Prompt templates
# -------------------------------

SYSTEM_CN = (
    "你是一个精炼的中文会议纪要助手。请将内容按结构化要点输出，"
    "包括：主题、时间/参与者（若有）、关键结论、行动项（含责任人/截止时间）、风险与未决问题。"
    "语言要简洁准确，保留数字、指标与专有名词。"
)

INSTRUCT_CN_MAP = (
    "请对以下内容进行结构化要点总结：\n"
    "- 主题与背景\n- 核心结论（尽可能量化）\n- 行动项（清单，含责任人/截止时间）\n- 风险与未决问题\n\n"
    "原文内容：\n{content}"
)

INSTRUCT_CN_REDUCE = (
    "下面是多个分块摘要，请综合去重并合并为一份完整、清晰、无矛盾的最终纪要：\n{content}\n\n"
    "要求：\n- 合并重复信息，补齐上下文\n- 保留数字/时间/人名/任务\n- 用中文分点输出，结构清晰"
)


# -------------------------------
# Utility
# -------------------------------

def read_text(p: Path) -> str:
    # Robust UTF-8 with fallback
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="utf-8", errors="replace")


def write_text(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.rstrip() + "\n", encoding="utf-8")


# -------------------------------
# Local summarizer abstraction
# -------------------------------

class LocalSummarizer:
    def __init__(
        self,
        model_name_or_path: str,
        backend: str = "transformers",
        load_cfg: Optional[LoadConfig] = None,
        gen_cfg: Optional[GenConfig] = None,
        chunk_cfg: Optional[ChunkConfig] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.backend = backend
        self.load_cfg = load_cfg or LoadConfig()
        self.gen_cfg = gen_cfg or GenConfig()
        self.chunk_cfg = chunk_cfg or ChunkConfig()

        if backend == "transformers":
            self._init_transformers()
        elif backend == "llama_cpp":
            self._init_llama_cpp()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # -------- Transformers backend --------
    def _init_transformers(self):
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        self.torch = torch
        self.AutoModelForCausalLM = AutoModelForCausalLM
        self.AutoTokenizer = AutoTokenizer
        self.BitsAndBytesConfig = BitsAndBytesConfig

        # Decide device
        device = self.load_cfg.device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # mac
                device = "mps"
            else:
                device = "cpu"
        self.device_kind = device

        # Try 4bit -> fp16/bf16 (GPU) -> cpu
        tokenizer = None
        model = None
        last_error = None

        # Always load tokenizer first (chat template may be needed for chunking)
        tokenizer = self.AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.load_cfg.trust_remote_code,
            use_fast=True,
        )

        # Ensure pad token exists to avoid warnings in generate
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        # Prepare kwargs
        common_kwargs: Dict[str, Any] = dict(
            trust_remote_code=self.load_cfg.trust_remote_code,
        )

        # Candidate strategies
        strategies = []

        def can_try_bnb() -> bool:
            if self.device_kind != "cuda":
                return False
            try:
                import bitsandbytes  # noqa: F401
                return True
            except Exception:
                return False

        # 1) 4bit on CUDA (preferred)
        if self.load_cfg.prefer_4bit and can_try_bnb():
            qcfg = self.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.torch.float16,  # IMPORTANT: torch dtype, not string
            )
            strategies.append(dict(
                name="cuda-4bit",
                kwargs=dict(
                    device_map="auto",
                    quantization_config=qcfg,
                )
            ))

        # 2) fp16/bf16 on GPU
        if self.device_kind == "cuda":
            dtype = self.torch.float16
            # Ampere+ can benefit from bfloat16; if you prefer, toggle here.
            strategies.append(dict(
                name="cuda-fp16",
                kwargs=dict(
                    device_map="auto",
                    torch_dtype=dtype,
                )
            ))

        # 3) CPU fallback
        strategies.append(dict(
            name="cpu-fp32",
            kwargs=dict(
                device_map="cpu",
            )
        ))

        for strat in strategies:
            try:
                model = self.AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    **common_kwargs,
                    **strat["kwargs"],
                )
                self._strategy = strat["name"]
                break
            except Exception as e:
                last_error = e
                # free cuda if partially allocated
                if self.device_kind == "cuda":
                    try:
                        self.torch.cuda.empty_cache()
                    except Exception:
                        pass
                continue

        if model is None:
            msg = f"All model load strategies failed. Last error: {last_error}"
            raise RuntimeError(msg)

        # Attach
        self.tokenizer = tokenizer
        self.model = model

        # Generation defaults
        from transformers import GenerationConfig
        gen = GenerationConfig(
            max_new_tokens=self.gen_cfg.max_new_tokens,
            temperature=self.gen_cfg.temperature,
            top_p=self.gen_cfg.top_p,
            do_sample=self.gen_cfg.do_sample,
            repetition_penalty=self.gen_cfg.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        self.model.generation_config = gen

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        # If tokenizer supports chat template, use it; otherwise, fallback.
        apply = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(apply):
            return apply(messages, tokenize=False, add_generation_prompt=True)
        # Basic fallback: simple concatenation
        out = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                out.append(f"[系统]\n{content}\n")
            elif role == "user":
                out.append(f"[用户]\n{content}\n")
            else:
                out.append(f"[{role}]\n{content}\n")
        out.append("答复：")
        return "\n".join(out).strip()

    def _gen(self, prompt: str) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt")
        # Move to model device (handles sharded with accelerate)
        if hasattr(self.model, "device"):
            device = self.model.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(**inputs)
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Try to strip the prompt prefix if template echos it
        if text.startswith(prompt):
            text = text[len(prompt):].lstrip()
        return text.strip()

    def summarize_text(self, content: str, system: str = SYSTEM_CN) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": INSTRUCT_CN_MAP.format(content=content)},
        ]
        prompt = self._apply_chat_template(messages)
        return self._gen(prompt)

    def reduce_summaries(self, partials: List[str]) -> str:
        joined = "\n\n---\n\n".join(partials)
        messages = [
            {"role": "system", "content": SYSTEM_CN},
            {"role": "user", "content": INSTRUCT_CN_REDUCE.format(content=joined)},
        ]
        prompt = self._apply_chat_template(messages)
        return self._gen(prompt)

    def count_tokens(self, text: str) -> int:
        # rough but reliable with fast tokenizer
        return len(self.tokenizer.encode(text))

    def chunk_by_tokens(self, text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
        # Token-aware chunking preserving boundaries roughly by sentence punctuation.
        # To keep it simple and robust, we do a sliding window over tokens.
        ids = self.tokenizer.encode(text)
        n = len(ids)
        if n <= max_tokens:
            return [text]

        chunks = []
        start = 0
        while start < n:
            end = min(n, start + max_tokens)
            piece_ids = ids[start:end]
            chunk = self.tokenizer.decode(piece_ids, skip_special_tokens=True)
            chunks.append(chunk.strip())
            if end == n:
                break
            start = max(0, end - overlap_tokens)
        return chunks

    # -------- llama.cpp backend --------
    def _init_llama_cpp(self):
        # For GGUF local models
        from llama_cpp import Llama

        # Heuristic defaults; adjust n_gpu_layers for your GPU if needed
        self.llm = Llama(
            model_path=self.model_name_or_path,
            n_ctx=8192,
            n_gpu_layers=60,  # tune this based on VRAM
            logits_all=False,
            chat_format="auto",  # use built-in if meta present
        )

    def _llama_chat(self, system: str, user: str) -> str:
        # Use llama_cpp's chat if available, otherwise plain prompt
        try:
            out = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.gen_cfg.temperature,
                top_p=self.gen_cfg.top_p,
                max_tokens=self.gen_cfg.max_new_tokens,
            )
            return out["choices"][0]["message"]["content"].strip()
        except Exception:
            prompt = f"<<SYS>>\n{system}\n<</SYS>>\n\n[INST] {user} [/INST]"
            out = self.llm(
                prompt=prompt,
                temperature=self.gen_cfg.temperature,
                top_p=self.gen_cfg.top_p,
                max_tokens=self.gen_cfg.max_new_tokens,
            )
            return out["choices"][0]["text"].strip()

    def summarize_text_llama(self, content: str, system: str = SYSTEM_CN) -> str:
        user = INSTRUCT_CN_MAP.format(content=content)
        return self._llama_chat(system, user)

    def reduce_summaries_llama(self, partials: List[str]) -> str:
        joined = "\n\n---\n\n".join(partials)
        user = INSTRUCT_CN_REDUCE.format(content=joined)
        return self._llama_chat(SYSTEM_CN, user)

    # -------- High-level map-reduce summarization --------
    def summarize_long(self, text: str) -> str:
        # Split -> map -> reduce
        if self.backend == "transformers":
            chunked = self.chunk_by_tokens(
                text,
                max_tokens=self.chunk_cfg.chunk_tokens,
                overlap_tokens=self.chunk_cfg.overlap_tokens,
            )
            partials = []
            for i, ch in enumerate(chunked, 1):
                try:
                    partial = self.summarize_text(ch)
                except Exception as e:
                    partial = f"[分块 {i} 摘要失败: {e}]"
                partials.append(partial)
            if len(partials) == 1:
                return partials[0]

            # Reduce stage with smaller joined input
            # If joined too long, reduce in stages
            joined = "\n\n---\n\n".join(partials)
            if self.count_tokens(joined) <= self.chunk_cfg.reduce_tokens:
                return self.reduce_summaries(partials)

            # Multi-stage reduce
            groups = []
            buf = []
            buf_tokens = 0
            for p in partials:
                t = self.count_tokens(p)
                if buf_tokens + t > self.chunk_cfg.reduce_tokens and buf:
                    groups.append(buf)
                    buf, buf_tokens = [], 0
                buf.append(p)
                buf_tokens += t
            if buf:
                groups.append(buf)

            mids = [self.reduce_summaries(g) for g in groups]
            return self.reduce_summaries(mids)

        elif self.backend == "llama_cpp":
            # Similar pipeline for llama
            # Simpler chunking by tokens is not available; use char window fallback
            units = self._char_chunks(text, max_chars=4000, overlap=300)
            partials = [self.summarize_text_llama(u) for u in units]
            if len(partials) == 1:
                return partials[0]
            return self.reduce_summaries_llama(partials)

        else:
            raise ValueError("Unsupported backend")

    def _char_chunks(self, text: str, max_chars: int, overlap: int) -> List[str]:
        if len(text) <= max_chars:
            return [text]
        out = []
        start = 0
        n = len(text)
        while start < n:
            end = min(n, start + max_chars)
            out.append(text[start:end].strip())
            if end == n:
                break
            start = max(0, end - overlap)
        return out


# -------------------------------
# Public function: summarize_file
# -------------------------------

def summarize_file(
    input_md_path: Path,
    output_md_path: Path,
    model_path: str,
    backend: str = "transformers",
    device: str = "auto",
    prefer_4bit: bool = True,
) -> None:
    """
    Summarize a *_merge.md file into *_summary.md using a local LLM.
    - input_md_path: source transcript markdown
    - output_md_path: target summary markdown
    - model_path: HF repo id (e.g. 'Qwen/Qwen2.5-3B-Instruct') or local dir (e.g. 'H:\\models\\Qwen2.5-3B-Instruct')
    - backend: 'transformers' | 'llama_cpp'
    - device: 'auto' | 'cuda' | 'cpu'
    - prefer_4bit: try 4bit (bitsandbytes) when possible
    """
    text = read_text(input_md_path)

    load_cfg = LoadConfig(device=device, prefer_4bit=prefer_4bit, trust_remote_code=True)
    gen_cfg = GenConfig(max_new_tokens=800, temperature=0.2, top_p=0.9, do_sample=False)
    chunk_cfg = ChunkConfig(chunk_tokens=1800, overlap_tokens=150, reduce_tokens=1600)

    # Instantiate
    summ = LocalSummarizer(
        model_name_or_path=model_path,
        backend=backend,
        load_cfg=load_cfg,
        gen_cfg=gen_cfg,
        chunk_cfg=chunk_cfg,
    )

    # Summarize
    try:
        summary = summ.summarize_long(text)
    except Exception as e:
        # Last-resort: print traceback and re-raise to help diagnose
        tb = traceback.format_exc()
        err_note = f"[Summarization failed]\n{e}\n\n{tb}"
        write_text(output_md_path, err_note)
        raise

    # Write result with small header
    header = (
        f"# 摘要\n\n"
        f"- **模型:** {model_path}\n"
        f"- **后端:** {backend}\n"
        f"- **加载策略:** {getattr(summ, '_strategy', 'n/a')}\n\n"
    )
    write_text(output_md_path, header + summary)
