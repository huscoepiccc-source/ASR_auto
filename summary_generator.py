import os
import glob
import requests
import json
import time

# 配置信息
INPUT_DIR = r"C:\Users\Administrator\Downloads\input"  # 输入目录
# API_KEY = "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"  # 替换为你的实际API密钥
# MODEL_NAME = "mistralai/mistral-7b-instruct:free"  # 推荐的免费模型
# API_URL = "https://openrouter.ai/api/v1/chat/completions"
# 使用DeepSeek的配置   先注释掉，以后再使用
#API_URL = " https://api.deepseek.com/v1/chat/completions"
#MODEL_NAME = "deepseek-chat"  # 另外一个是 deepseek-reasoner  官方放出来的名称。总结比较长的文本时候使用。
#API_KEY = "sk-d5dccc7fc4ba48bfb451f013779eaccf"  #

from ctransformers import AutoModelForCausalLM

# 加载开源模型（首次运行会自动下载）
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    model_type="mistral"
)

def local_generate_summary(text_content):
    prompt = f"{SYSTEM_PROMPT}\n\n请总结以下文本：{text_content}"
    response = model(prompt, max_new_tokens=500)
    return response



# 系统提示词 - 定义AI的角色和任务要求
SYSTEM_PROMPT = """
你是一个专业的文本摘要生成器。你的任务是根据用户提供的文本内容，生成结构化的摘要。
请严格按照以下要求生成摘要：
1. 生成一个5-20字的标题
2. 用一句话概括文本的核心内容
3. 文章标题：准确反映文本主题
4. 文本梗概：列出3-5个主要内容要点
5. 文本知识要点：分点列出关键知识点（不超过5点）
6. 整体摘要控制在300字以内
7. 输出格式使用Markdown，包含二级标题和列表
"""

def read_md_file(file_path):
    """读取Markdown文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None

def generate_summary(text_content):
    """调用OpenRouter API生成摘要"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"请总结以下文本：\n\n{text_content}"}
        ],
        "temperature": 0.3,  # 较低的随机性保证结果稳定
        "max_tokens": 500    # 限制输出长度
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # 检查HTTP错误
        
        result = response.json()
        return result['choices'][0]['message']['content']
    
    except requests.exceptions.RequestException as e:
        print(f"API请求失败: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"API响应解析错误: {e}")
        return None

def save_summary(original_file, original_content, summary):
    """保存摘要到新文件"""
    base_name = os.path.splitext(original_file)[0]
    output_file = os.path.join(INPUT_DIR, f"{base_name}_summary.md")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # 写入原始内容
            f.write(f"# 原始文本\n\n")
            f.write(original_content)
            f.write("\n\n---\n\n")
            
            # 写入摘要
            f.write(f"# 文本摘要\n\n")
            f.write(summary)
        
        print(f"摘要已保存到: {output_file}")
        return True
    except Exception as e:
        print(f"保存文件失败: {e}")
        return False

def process_files():
    """处理目录中的所有Markdown文件"""
    # 获取目录中所有.md文件（排除已生成的摘要文件）
    md_files = glob.glob(os.path.join(INPUT_DIR, "*.md"))
    md_files = [f for f in md_files if "_summary.md" not in f]
    
    if not md_files:
        print(f"在 {INPUT_DIR} 中未找到Markdown文件")
        return
    
    print(f"找到 {len(md_files)} 个待处理文件")
    
    for file_path in md_files:
        file_name = os.path.basename(file_path)
        print(f"\n处理文件: {file_name}")
        
        # 读取文件内容
        original_content = read_md_file(file_path)
        if not original_content:
            continue
        
        # 调用API生成摘要
        print("正在生成摘要...")
        summary = generate_summary(original_content)
        
        if not summary:
            print("摘要生成失败，跳过此文件")
            continue
        
        # 保存结果
        if save_summary(file_name, original_content, summary):
            print("处理成功！")
        
        # 避免API速率限制（免费账号建议间隔）
        time.sleep(3)

if __name__ == "__main__":
    process_files()
    print("\n所有文件处理完成！")