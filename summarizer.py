# -*- coding: utf-8 -*-
"""
Simplest stable summarizer for Qwen2.5-3B-Instruct
- transformers==4.55.0, accelerate==1.10.0
- Local model dir: e.g. H:\\models\\Qwen2.5-3B-Instruct
- Strategy: CUDA 4bit (bnb) -> CUDA FP16 -> CPU
- Single-pass generation, no chunking
"""

from pathlib import Path
from typing import Tuple, Any

SYSTEM_PROMPT = (
    "你是一个高效的中文纪要助手。请对给定转写内容进行简明而结构化的总结。"
    "输出要求：\n"
    "1) 用 3-5 个要点做速览（不超过 200 字）\n"
    "2) 给出分级大纲（使用 Markdown 标题 H2/H3）\n"
    "3) 列出关键要点（数据/结论/证据）\n"
    "4) 若有行动项，请用 [负责人/截止日期/依赖] 格式列出\n"
    "保持客观，不臆测，必要时标注“不确定之处”。"
)
USER_PROMPT_TEMPLATE = (
    "请对以下内容进行总结与结构化提炼：\n\n"
    "{content}\n\n"
    "请按上述要求输出，使用 Markdown。"
)

MAX_NEW_TOKENS = 256
MIN_NEW_TOKENS = 64       # 防止 0-token 输出
REPETITION_PENALTY = 1.05
TRUNCATE_AT_CHARS = 8000  # 防止极端长文本导致拖慢/溢出


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _bnb_ok() -> bool:
    try:
        import bitsandbytes  # noqa: F401
        return True
    except Exception:
        return False


def _load_qwen(model_dir: str) -> Tuple[Any, Any, str]:
    """
    Returns tokenizer, model, strategy
    strategy in {"cuda-4bit", "cuda-fp16", "cpu-fp32"}
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    dev = _device()
    last_err = None

    # Try CUDA 4bit
    if dev == "cuda" and _bnb_ok():
        try:
            qcfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,  # torch.dtype 必须
            )
            mdl = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                device_map="auto",
                quantization_config=qcfg,
            )
            return tok, mdl, "cuda-4bit"
        except Exception as e:
            last_err = e
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    # Try CUDA FP16
    if dev == "cuda":
        try:
            mdl = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            return tok, mdl, "cuda-fp16"
        except Exception as e:
            last_err = e
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    # CPU fallback
    try:
        mdl = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            device_map="cpu",
        )
        return tok, mdl, "cpu-fp32"
    except Exception as e:
        last_err = e
        raise RuntimeError(f"Load model failed: {last_err}")


def _chat_to_tensors(tokenizer, system: str, user: str, device: str):
    """
    使用 chat 模板直接返回 tensor，避免二次分词导致前后缀对不上。
    """
    has_template = callable(getattr(tokenizer, "apply_chat_template", None))
    if has_template:
        inputs = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        # 回退到纯串接
        text = f"[系统]\n{system}\n\n[用户]\n{user}\n\n答复："
        inputs = tokenizer(text, return_tensors="pt").input_ids
    return inputs.to(device)


def _gen_once(tokenizer, model, input_ids, min_new=MIN_NEW_TOKENS, do_sample=False, temperature=0.7):
    """
    只解码新增 token：outputs[:, input_len:]
    """
    import torch
    input_len = input_ids.shape[-1]
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            min_new_tokens=min_new,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=0.9 if do_sample else None,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_ids = outputs[:, input_len:]
    text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    return text


def summarize_text(content: str, model_dir: str) -> str:
    # 简单截断，防止异常长文本
    if len(content) > TRUNCATE_AT_CHARS:
        content = content[:TRUNCATE_AT_CHARS] + "\n\n[... 已截断 ...]"

    tokenizer, model, strategy = _load_qwen(model_dir)

    # 构造输入
    user = USER_PROMPT_TEMPLATE.format(content=content)
    device = getattr(model, "device", _device())
    input_ids = _chat_to_tensors(tokenizer, SYSTEM_PROMPT, user, device)

    # 第一次尝试：贪心 + min_new_tokens 防止 0 token
    text = _gen_once(tokenizer, model, input_ids, min_new=MIN_NEW_TOKENS, do_sample=False)

    # 兜底：如果仍然空，改用采样再试一次
    if not text:
        text = _gen_once(tokenizer, model, input_ids, min_new=MIN_NEW_TOKENS, do_sample=True, temperature=0.7)

    # 释放
    try:
        import torch, gc
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    header = f"# 摘要\n\n- **模型:** {model_dir}\n- **加载策略:** {strategy}\n\n"
    return header + (text if text else "_[生成为空]_ 请重试或增大 MIN_NEW_TOKENS/调整温度。")


def summarize_file(input_md_path: Path, output_md_path: Path, model_path: str, backend: str = "transformers") -> None:
    if backend != "transformers":
        raise ValueError("This simple version supports 'transformers' only.")
    text = _read_text(input_md_path)
    summary = summarize_text(text, model_path)
    _write_text(output_md_path, summary)


def _read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="utf-8", errors="replace")


def _write_text(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.rstrip() + "\n", encoding="utf-8")
