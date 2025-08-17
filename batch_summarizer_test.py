from pathlib import Path
from llama_cpp import Llama

# 模型路径（GGUF）
MODEL_PATH = r"H:\Projects\AudioSeparationGUI\asr_env\.cache\models\Qwen\Qwen3-4B-GGUF\Qwen3-4B-Q4_K_M.gguf"
# 输入目录
INPUT_DIR = Path(r"C:\Users\Administrator\Downloads\input")

# 固定提示词（你的结构化摘要模板）
PROMPT_TEMPLATE = """你是一个专业的文本摘要生成器。你的任务是根据用户提供的文本内容，生成结构化的摘要。
请严格按照以下要求生成摘要：
1. 标题：生成一个5-20字的标题，准确反映文本主题
2. 核心内容：用一句话概括文本的
3. 文本梗概：列出3-7个主要内容要点
4. 文本知识要点：分点列出关键知识点（不超过7点）
5. 整体摘要控制在800字以内
6. 非常重要的一点，书写格式请务必要严格要求，因为后续我会拿来进行批量替换，所以务必严格遵循我的要求。

# 文本摘要
# 标题
(此处为标题内容)

## 核心内容
(此处为核心内容)

### 文本梗概
(此处为文本梗概)

### 文本知识要点
(此处为文本知识要点)

注意：#号和中文字符之间 中间有一个空格符号。
请再次注意，此书写格式务必严格遵循!!!!

以下是需要总结的文本内容：
"""

# 加载模型
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=32768,
    n_threads=8,
    n_gpu_layers=-1
)

def summarize_file(md_path: Path):
    text_content = md_path.read_text(encoding="utf-8", errors="ignore")
    prompt = PROMPT_TEMPLATE + text_content

    output = llm(
        prompt,
        max_tokens=2048,
        temperature=0.6,
        top_p=0.95,
        stop=[]
    )

    summary_text = output["choices"][0]["text"].strip()

    # 写入新文件
    output_path = md_path.with_name(md_path.stem + "_offline_summary.md")
    output_path.write_text(
        f"# 原文内容\n{text_content}\n\n# AI总结\n{summary_text}",
        encoding="utf-8"
    )
    print(f"已处理: {md_path.name} -> {output_path.name}")

if __name__ == "__main__":
    md_files = list(INPUT_DIR.glob("*.md"))
    for file in md_files:
        summarize_file(file)
