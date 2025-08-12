# -*- coding: utf-8 -*-
import sys
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

# ========== 项目相对路径探测 ==========
PROJECT_ROOT = Path(__file__).resolve().parent

# 虚拟环境 python.exe（Windows）
VENV_PY = PROJECT_ROOT / "asr_env" / "Scripts" / "python.exe"

# batch_runner.py 位置（兼容 executive / excutive）
def find_batch_runner():
    candidates = [
        PROJECT_ROOT / "executive" / "batch_runner.py",
        PROJECT_ROOT / "excutive" / "batch_runner.py",
        PROJECT_ROOT / "batch_runner.py",  # 兜底：就在根目录的情况
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

BATCH_RUNNER = find_batch_runner()

# 可选：默认的 model-dir（按你的离线缓存相对位置）
DEFAULT_MODEL_DIR = PROJECT_ROOT / "asr_env" / ".cache" / "modelscope" / "hub" / "iic"


# ========== GUI ==========

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("批量转录启动器")
        self.geometry("700x260")

        # 输入目录
        tk.Label(self, text="输入目录:").grid(row=0, column=0, sticky="e", padx=8, pady=8)
        self.input_var = tk.StringVar()
        self.input_entry = tk.Entry(self, textvariable=self.input_var, width=70)
        self.input_entry.grid(row=0, column=1, padx=8, pady=8, sticky="we")
        tk.Button(self, text="浏览...", command=self.choose_input).grid(row=0, column=2, padx=8, pady=8)

        # 输出目录（可选，不填则让 batch_runner 用默认或自己映射）
        tk.Label(self, text="输出目录(可选):").grid(row=1, column=0, sticky="e", padx=8, pady=8)
        self.output_var = tk.StringVar()
        self.output_entry = tk.Entry(self, textvariable=self.output_var, width=70)
        self.output_entry.grid(row=1, column=1, padx=8, pady=8, sticky="we")
        tk.Button(self, text="浏览...", command=self.choose_output).grid(row=1, column=2, padx=8, pady=8)

        # model-dir（可选）
        tk.Label(self, text="模型目录(可选):").grid(row=2, column=0, sticky="e", padx=8, pady=8)
        self.model_var = tk.StringVar(value=str(DEFAULT_MODEL_DIR))
        self.model_entry = tk.Entry(self, textvariable=self.model_var, width=70)
        self.model_entry.grid(row=2, column=1, padx=8, pady=8, sticky="we")
        tk.Button(self, text="浏览...", command=self.choose_model).grid(row=2, column=2, padx=8, pady=8)

        # 设备选择、递归、覆写
        tk.Label(self, text="设备:").grid(row=3, column=0, sticky="e", padx=8, pady=8)
        self.device_var = tk.StringVar(value="cuda")
        device_box = tk.OptionMenu(self, self.device_var, "cuda", "cpu")
        device_box.config(width=8)
        device_box.grid(row=3, column=1, sticky="w", padx=8, pady=8)

        self.recursive_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self, text="递归子目录", variable=self.recursive_var).grid(row=3, column=1, padx=110, pady=8, sticky="w")

        self.overwrite_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self, text="覆盖已存在结果", variable=self.overwrite_var).grid(row=3, column=1, padx=220, pady=8, sticky="w")

        # 开始按钮
        self.start_btn = tk.Button(self, text="开始转录", command=self.start, bg="#2e7d32", fg="white")
        self.start_btn.grid(row=4, column=1, padx=8, pady=16, sticky="e")

        # 约束列权重让中间列伸缩
        self.grid_columnconfigure(1, weight=1)

        # 预运行检查
        self.preflight_check()

    def preflight_check(self):
        missing = []
        if not VENV_PY.exists():
            missing.append(f"未找到虚拟环境 Python: {VENV_PY}")
        if BATCH_RUNNER is None:
            missing.append("未找到 batch_runner.py（尝试了 executive/、excutive/、项目根目录）")
        if missing:
            messagebox.showerror("环境检查失败", "\n".join(missing))

    def choose_input(self):
        d = filedialog.askdirectory(title="选择音视频所在的输入目录")
        if d:
            self.input_var.set(d)

    def choose_output(self):
        d = filedialog.askdirectory(title="选择输出目录（可选）")
        if d:
            self.output_var.set(d)

    def choose_model(self):
        d = filedialog.askdirectory(title="选择本地模型目录（可选）")
        if d:
            self.model_var.set(d)

    def start(self):
        if BATCH_RUNNER is None or not VENV_PY.exists():
            messagebox.showerror("无法启动", "环境未就绪，请检查虚拟环境和 batch_runner.py 路径。")
            return

        input_dir = self.input_var.get().strip()
        if not input_dir:
            messagebox.showwarning("缺少输入目录", "请选择输入目录。")
            return
        if not Path(input_dir).exists():
            messagebox.showwarning("路径不存在", f"输入目录不存在：{input_dir}")
            return

        # 组装命令行参数
        cmd = [str(VENV_PY), str(BATCH_RUNNER), "--input", input_dir]

        output_dir = self.output_var.get().strip()
        if output_dir:
            cmd += ["--output", output_dir]

        model_dir = self.model_var.get().strip()
        if model_dir:
            cmd += ["--model-dir", model_dir]

        device = self.device_var.get()
        if device:
            cmd += ["--device", device]

        if self.recursive_var.get():
            cmd += ["--recursive"]
        if self.overwrite_var.get():
            cmd += ["--overwrite"]

        # 设置工作目录为 batch_runner.py 所在目录，保证相对导入正常
        cwd = Path(BATCH_RUNNER).parent

        # 启动子进程（显示控制台窗口，便于看日志；如果不想要控制台，可以改 creationflags）
        try:
            subprocess.Popen(
                cmd,
                cwd=str(cwd),
                shell=False
            )
            messagebox.showinfo("已启动", "转录任务已启动。\n请查看控制台窗口或输出目录。")
        except Exception as e:
            messagebox.showerror("启动失败", str(e))


if __name__ == "__main__":
    app = App()
    app.mainloop()
