# summarizer.py
from pathlib import Path
import re
import math

def chunk_text(text, max_chars=3000):
    # 按段落/句号粗切，避免打断语义
    parts, buf = [], []
    size = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            line = "\n"
        ln = len(line)
        if size + ln > max_chars and buf:
            parts.append("\n".join(buf).strip())
            buf, size = [line], ln
        else:
            buf.append(line)
            size += ln
    if buf:
        parts.append("\n".join(buf).strip())
    return parts

DEFAULT_PROMPT = """你是中文文本编辑与信息架构专家。请对以下转写内容进行高质量的提炼与结构化输出：
目标：
- 提炼核心观点、关键信息、决策与行动项
- 以 Markdown 输出：先给 150~250 字「速览摘要」，再给「分级大纲」、再给「要点清单」、最后给「行动项」
- 保持客观，不臆测；必要时给出「不确定之处」
格式要求（严格遵守）：
# 速览
- 用 3~5 条 bullet，总字数 150~250

# 大纲
- 使用 H2/H3 分级标题归纳主题与层次

# 要点
- 用 bullet 提炼数据、结论、证据或引用

# 行动项
- 用 [负责人/截止日期/依赖] 标注；无法判定则写「待定」
"""

REDUCE_PROMPT = """你将看到若干分块的“摘要结果（分块）”。请在不丢失关键信息的前提下合并为一份最终稿。
要求：同样遵守上文的「格式要求」，保持结构清晰，去重、合并相似点，标注不确定之处。
"""

class LocalSummarizer:
    def __init__(self, model_name_or_path: str, backend="transformers", device="cuda", load_4bit=True):
        self.backend = backend
        if backend == "transformers":
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            kwargs = {}
            if load_4bit:
                kwargs.update(dict(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16"))
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, trust_remote_code=True, device_map="auto", **kwargs
            )
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=800,
                do_sample=False,
                temperature=0.2,
                top_p=0.9
            )
        elif backend == "llama_cpp":
            from llama_cpp import Llama
            # GGUF 模型路径，例如: ./models/qwen2.5-3b-instruct-q4_k_m.gguf
            self.llm = Llama(model_path=model_name_or_path, n_ctx=8192, n_gpu_layers=60)

    def _gen(self, prompt: str):
        if self.backend == "transformers":
            out = self.pipe(prompt)[0]["generated_text"]
            return out[len(prompt):].strip()
        else:
            out = self.llm(prompt=prompt, max_tokens=900, temperature=0.2, top_p=0.9)
            return out["choices"][0]["text"].strip()

    def summarize_long(self, raw_text: str):
        chunks = chunk_text(raw_text, max_chars=3500)
        partials = []
        for i, c in enumerate(chunks, 1):
            p = f"{DEFAULT_PROMPT}\n\n[分块 {i}/{len(chunks)}]\n\n{c}\n\n请输出："
            partials.append(self._gen(p))
        merged_input = "\n\n".join(f"[分块稿件 {i+1}]\n{t}" for i, t in enumerate(partials))
        final_prompt = f"{DEFAULT_PROMPT}\n\n{REDUCE_PROMPT}\n\n{merged_input}\n\n请输出最终稿："
        return self._gen(final_prompt)

def summarize_file(md_path: Path, out_path: Path, model_path: str, backend="transformers"):
    text = Path(md_path).read_text(encoding="utf-8")
    summ = LocalSummarizer(model_path, backend=backend, device="cuda", load_4bit=True)
    result = summ.summarize_long(text)
    out_path.write_text(result, encoding="utf-8")
    return out_path
