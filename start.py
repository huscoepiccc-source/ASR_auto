# -*- coding: utf-8 -*-
"""
批量转录启动器（拖拽目录 + 预置/自定义脚本二选一）
- 预置格式（executive / excutive 下的脚本）
- 自定义脚本（任意 .py）
- 二选一互斥，另一侧自动禁用
- 可选：拖拽目录/文件到“输入目录”框（若安装了 tkinterdnd2）
"""

import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

# ========== 可选：拖拽支持（tkinterdnd2） ==========
DND_AVAILABLE = False
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except Exception:
    DND_AVAILABLE = False

# 根据是否可用 DnD 决定根窗口基类
BaseTk = TkinterDnD.Tk if DND_AVAILABLE else tk.Tk

# ========== 项目路径与常量 ==========
PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PY = PROJECT_ROOT / "asr_env" / "Scripts" / "python.exe"

EXEC_DIR_CANDIDATES = [
    PROJECT_ROOT / "executive",
    PROJECT_ROOT / "excutive",
]

def get_exec_dir() -> Path | None:
    for d in EXEC_DIR_CANDIDATES:
        if d.exists():
            return d
    return None

# 预置脚本映射（名字 -> 文件名）
SCRIPT_OPTIONS = {
    "txt类型_有date": "batch_runner_merge.date.txt.py",
    "txt类型_无date": "batch_runner_merge.txt.py",
    "md_v1_No_format": "batch_runner_merge.md_v1_No_format.py",
    "md_v2_Md_format": "batch_runner_merge.md_v2_Md_format.py",
    "srt,merge.txt": "batch_runner_srt,merge.txt.py",
}

DEFAULT_MODEL_DIR = PROJECT_ROOT / "asr_env" / ".cache" / "modelscope" / "hub" / "iic"

# ========== GUI ==========
class App(BaseTk):
    def __init__(self):
        super().__init__()
        self.title("批量转录启动器（拖拽目录 / 预置-自定义二选一）")
        self.geometry("820x380")
        self._build_widgets()
        self.grid_columnconfigure(1, weight=1)
        self.preflight_check()
        if DND_AVAILABLE:
            self._enable_dnd()

    # ---------- UI 构建 ----------
    def _build_widgets(self):
        row = 0

        # 输入目录
        tk.Label(self, text="输入目录:").grid(row=row, column=0, sticky="e", padx=8, pady=8)
        self.input_var = tk.StringVar()
        self.input_entry = tk.Entry(self, textvariable=self.input_var, width=85)
        self.input_entry.grid(row=row, column=1, padx=8, pady=8, sticky="we")
        tk.Button(self, text="浏览...", command=self.choose_input).grid(row=row, column=2, padx=8, pady=8)
        row += 1

        # 拖拽提示
        tip_text = "提示：可将文件夹/文件拖入上方输入框" if DND_AVAILABLE else "提示：拖拽未启用（可选安装 tkinterdnd2）"
        self.dnd_tip = tk.Label(self, text=tip_text, fg="#666666")
        self.dnd_tip.grid(row=row, column=1, sticky="w", padx=8, pady=(0, 8))
        row += 1

        # 输出目录
        tk.Label(self, text="输出目录(可选):").grid(row=row, column=0, sticky="e", padx=8, pady=8)
        self.output_var = tk.StringVar()
        tk.Entry(self, textvariable=self.output_var, width=85).grid(row=row, column=1, padx=8, pady=8, sticky="we")
        tk.Button(self, text="浏览...", command=self.choose_output).grid(row=row, column=2, padx=8, pady=8)
        row += 1

        # 模型目录
        tk.Label(self, text="模型目录(可选):").grid(row=row, column=0, sticky="e", padx=8, pady=8)
        self.model_var = tk.StringVar(value=str(DEFAULT_MODEL_DIR))
        tk.Entry(self, textvariable=self.model_var, width=85).grid(row=row, column=1, padx=8, pady=8, sticky="we")
        tk.Button(self, text="浏览...", command=self.choose_model).grid(row=row, column=2, padx=8, pady=8)
        row += 1

        # —— 格式选择分组（互斥）——
        group = tk.LabelFrame(self, text="转录格式选择（二选一）")
        group.grid(row=row, column=0, columnspan=3, padx=8, pady=8, sticky="we")
        group.grid_columnconfigure(1, weight=1)
        row += 1

        # 模式单选：preset or custom
        self.mode_var = tk.StringVar(value="preset")
        tk.Radiobutton(group, text="预置转录格式", variable=self.mode_var, value="preset",
                       command=self.update_mode).grid(row=0, column=0, sticky="w", padx=8, pady=6)
        tk.Radiobutton(group, text="采用其他格式（选择自定义 .py）", variable=self.mode_var, value="custom",
                       command=self.update_mode).grid(row=0, column=1, sticky="w", padx=8, pady=6)

        # 预置格式控件
        tk.Label(group, text="预置格式:").grid(row=1, column=0, sticky="e", padx=8, pady=6)
        self.preset_var = tk.StringVar(value=list(SCRIPT_OPTIONS.keys())[0])
        self.preset_menu = tk.OptionMenu(group, self.preset_var, *SCRIPT_OPTIONS.keys())
        self.preset_menu.config(width=32)
        self.preset_menu.grid(row=1, column=1, sticky="w", padx=8, pady=6)

        # 自定义脚本控件
        tk.Label(group, text="自定义脚本:").grid(row=2, column=0, sticky="e", padx=8, pady=6)
        self.custom_script_var = tk.StringVar()
        self.custom_entry = tk.Entry(group, textvariable=self.custom_script_var, width=70)
        self.custom_entry.grid(row=2, column=1, padx=8, pady=6, sticky="we")
        self.custom_btn = tk.Button(group, text="浏览 .py", command=self.choose_custom_script)
        self.custom_btn.grid(row=2, column=2, padx=8, pady=6)

        # 初始化模式禁用状态
        self.update_mode()

        # 设备、递归、覆写
        tk.Label(self, text="设备:").grid(row=row, column=0, sticky="e", padx=8, pady=8)
        self.device_var = tk.StringVar(value="cuda")
        device_box = tk.OptionMenu(self, self.device_var, "cuda", "cpu")
        device_box.config(width=10)
        device_box.grid(row=row, column=1, sticky="w", padx=8, pady=8)

        self.recursive_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self, text="递归子目录", variable=self.recursive_var).grid(row=row, column=1, padx=140, pady=8, sticky="w")

        self.overwrite_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self, text="覆盖已存在结果", variable=self.overwrite_var).grid(row=row, column=1, padx=260, pady=8, sticky="w")
        row += 1

        # 启动按钮
        self.start_btn = tk.Button(self, text="开始转录", command=self.start, bg="#2e7d32", fg="white")
        self.start_btn.grid(row=row, column=1, padx=8, pady=16, sticky="e")

    # ---------- 启用拖拽 ----------
    def _enable_dnd(self):
        # 将输入框注册为拖拽目标
        try:
            self.input_entry.drop_target_register(DND_FILES)  # type: ignore[attr-defined]
            self.input_entry.dnd_bind("<<Drop>>", self._on_drop_input)  # type: ignore[attr-defined]
        except Exception:
            # 极少数情况下主题/packaging导致方法缺失，直接忽略，保持正常功能
            pass

    def _on_drop_input(self, event):
        """
        处理拖拽数据：
        - 使用 tk.splitlist 解析可能的多项 & 带空格路径
        - 取第一项；若是文件则取其父目录；若是目录直接使用
        """
        try:
            items = self.tk.splitlist(event.data)  # 处理 {C:\path with space} 形式
        except Exception:
            items = [event.data]

        chosen = None
        for raw in items:
            p = Path(raw.strip().strip("{}")).resolve()
            if p.is_dir():
                chosen = p
                break
            if p.is_file():
                chosen = p.parent
                break

        if chosen:
            self.input_var.set(str(chosen))

    # ---------- 交互逻辑 ----------
    def set_widgets_state(self, widgets, state: str):
        for w in widgets:
            try:
                w.configure(state=state)
            except Exception:
                pass  # Label 等无 state

    def update_mode(self):
        mode = self.mode_var.get()
        if mode == "preset":
            self.set_widgets_state([self.preset_menu], "normal")
            self.set_widgets_state([self.custom_entry, self.custom_btn], "disabled")
        else:
            self.set_widgets_state([self.preset_menu], "disabled")
            self.set_widgets_state([self.custom_entry, self.custom_btn], "normal")

    # ---------- 选择器 ----------
    def choose_input(self):
        d = filedialog.askdirectory(title="选择输入目录")
        if d:
            self.input_var.set(d)

    def choose_output(self):
        d = filedialog.askdirectory(title="选择输出目录（可选）")
        if d:
            self.output_var.set(d)

    def choose_model(self):
        d = filedialog.askdirectory(title="选择模型目录（可选）")
        if d:
            self.model_var.set(d)

    def choose_custom_script(self):
        f = filedialog.askopenfilename(
            title="选择自定义 Python 脚本",
            filetypes=[("Python 脚本", "*.py"), ("所有文件", "*.*")]
        )
        if f:
            self.custom_script_var.set(f)

    # ---------- 预运行检查 ----------
    def preflight_check(self):
        notes = []
        if not VENV_PY.exists():
            notes.append(f"未找到虚拟环境 Python: {VENV_PY}")
        if get_exec_dir() is None:
            notes.append("未找到 executive / excutive 目录（仅影响“预置格式”）")
        if not DND_AVAILABLE:
            notes.append("拖拽未启用：可选安装 tkinterdnd2 以启用拖拽到输入框")
        if notes:
            messagebox.showwarning("环境提示", "\n".join(notes))

    # ---------- 启动 ----------
    def start(self):
        # 输入目录校验
        input_dir = self.input_var.get().strip()
        if not input_dir:
            messagebox.showwarning("缺少输入目录", "请选择输入目录或拖拽文件夹到输入框。")
            return
        if not Path(input_dir).exists():
            messagebox.showwarning("路径不存在", f"输入目录不存在：{input_dir}")
            return

        # 选择脚本路径
        mode = self.mode_var.get()
        if mode == "preset":
            exec_dir = get_exec_dir()
            if exec_dir is None:
                messagebox.showerror("缺少目录", "未找到 executive / excutive 目录，无法使用预置格式。")
                return
            script_name = SCRIPT_OPTIONS.get(self.preset_var.get())
            runner_path = exec_dir / script_name
        else:
            runner_path = Path(self.custom_script_var.get().strip())
            if not runner_path:
                messagebox.showwarning("缺少脚本", "请选择自定义 .py 脚本。")
                return

        if not runner_path.exists():
            messagebox.showerror("脚本不存在", f"找不到脚本文件:\n{runner_path}")
            return
        if runner_path.suffix.lower() != ".py":
            messagebox.showerror("类型不符", "请选择 .py 脚本文件。")
            return
        if not VENV_PY.exists():
            messagebox.showerror("虚拟环境错误", f"未找到虚拟环境 Python：\n{VENV_PY}")
            return

        # 组装命令
        cmd = [str(VENV_PY), str(runner_path), "--input", input_dir]

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
            cmd.append("--recursive")
        if self.overwrite_var.get():
            cmd.append("--overwrite")

        try:
            subprocess.Popen(cmd, cwd=str(runner_path.parent), shell=False)
            messagebox.showinfo("已启动", "转录任务已启动，请查看控制台窗口或输出目录。")
        except Exception as e:
            messagebox.showerror("启动失败", str(e))

if __name__ == "__main__":
    app = App()
    app.mainloop()
