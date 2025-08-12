import argparse
import concurrent.futures as futures
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple

from pipeline import InferencePipeline  # 确保与 batch_runner.py 同目录，或已在 PYTHONPATH

SUPPORTED_AUDIO = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".wma"}
SUPPORTED_VIDEO = {".mp4", ".mkv", ".mov", ".avi", ".flv", ".webm"}
SUPPORTED_ALL = SUPPORTED_AUDIO | SUPPORTED_VIDEO


def has_ffmpeg() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return True
    except Exception:
        return False


def ensure_wav(input_path: Path, work_dir: Path) -> Path:
    """
    若是 wav 直接返回；否则优先用 ffmpeg 转成临时 16k 单声道 wav。
    无 ffmpeg 时直接返回原文件（要求 pipeline 自行处理非 wav / 视频）。
    """
    if input_path.suffix.lower() in SUPPORTED_AUDIO and input_path.suffix.lower() == ".wav":
        return input_path

    if not has_ffmpeg():
        return input_path

    out = work_dir / f"{input_path.stem}.__tmp__.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def format_srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def write_srt(segments: List[Dict[str, Any]], out_path: Path):
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = format_srt_time(float(seg.get("start", 0.0)))
        end = format_srt_time(float(seg.get("end", 0.0)))
        speaker = seg.get("speaker", "")
        spk_prefix = f"{speaker}: " if speaker else ""
        text = seg.get("text", "")
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(f"{spk_prefix}{text}".strip())
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")

def write_txt(full_text: str, out_path: Path):
    out_path.write_text(full_text.strip() + "\n", encoding="utf-8")

def write_json(payload: Dict[str, Any], out_path: Path):
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def write_merge_txt(segments: List[Dict[str, Any]], out_path: Path, min_chars: int = 200, max_chars: int = 250):
    """
    将相邻同一说话人的片段合并成段；若一段过长，以句号“。”优先切分，每段控制在 200–250 字。
    输出格式：
    speaker 0
    这一段合并后的文本（可被拆成多个 200-250 字段落，每段之间空一行）

    speaker 1
    ...
    """
    def split_into_paragraphs(text: str) -> List[str]:
        # 保留句号作为边界；按“。”拆句，再按 min/max 长度聚合
        sentences = []
        buf = []
        for ch in text:
            buf.append(ch)
            if ch == "。":
                s = "".join(buf).strip()
                if s:
                    sentences.append(s)
                buf = []
        tail = "".join(buf).strip()
        if tail:
            sentences.append(tail)

        paras = []
        cur = ""
        for s in sentences:
            if not cur:
                cur = s
                continue
            # 若当前段不足 min_chars，则尽量继续累加；超过 max_chars 就换段
            if len(cur) < min_chars or (len(cur) + len(s) <= max_chars):
                cur += s
            else:
                paras.append(cur)
                cur = s
        if cur:
            paras.append(cur)
        return paras

    lines: List[str] = []
    last_speaker = None
    buffer = []

    def flush_block():
        nonlocal buffer, last_speaker
        if not buffer:
            return
        block_text = "".join(buffer).strip()
        if block_text:
            lines.append(f"speaker {last_speaker}")
            for para in split_into_paragraphs(block_text):
                # 段落正文
                lines.append(para.strip())
                # 段落间空一行
                lines.append("")
        buffer = []

    for seg in segments:
        spk = seg.get("speaker", "UNK")  # 可能是 int/str，都允许
        text = (seg.get("text", "") or "").strip()
        # 碰到新说话人，先冲刷上一位说话人的合并段
        if last_speaker is None:
            last_speaker = spk
        if spk != last_speaker:
            flush_block()
            last_speaker = spk
        buffer.append(text)

    # 冲刷最后一段
    flush_block()

    out = "\n".join(lines).rstrip() + "\n"
    out_path.write_text(out, encoding="utf-8")

def out_path_merge_inplace(file_path: Path) -> Path:
    """在源目录就地生成 *_merge.txt"""
    return file_path.with_name(file_path.stem + "_merge.txt")

def out_path_merge_mirror(input_base: Path, output_base: Path, file_path: Path) -> Path:
    """在输出根路径进行目录镜像，生成 *_merge.txt"""
    rel = file_path.relative_to(input_base)
    target = output_base / rel
    return target.with_name(target.stem + "_merge.txt")


def mirror_output_path(input_base: Path, output_base: Path, file_path: Path, suffix: str) -> Path:
    rel = file_path.relative_to(input_base)
    target = output_base / rel
    return target.with_suffix(suffix)

def write_all_outputs(result: Dict[str, Any], base_path: Path):
    segments = result.get("segments", [])
    full_text = result.get("text", "")
    payload = result

    write_srt(segments, base_path.with_suffix(".srt"))
    write_txt(full_text, base_path.with_suffix(".txt"))
    write_json(payload, base_path.with_suffix(".json"))
    write_merge_txt(segments, base_path.with_name(base_path.stem + "_merge.txt"))


def process_one(file_path: Path, args, pipe: InferencePipeline) -> Tuple[Path, bool, str]:
    """
    返回：(srt_path, success, msg)
    """
    try:
        # 输出路径（同名不同后缀）
        # srt_path = mirror_output_path(args.input, args.output, file_path, ".srt")
        # txt_path = mirror_output_path(args.input, args.output, file_path, ".txt")
        # json_path = mirror_output_path(args.input, args.output, file_path, ".json")
        # tmp_work = srt_path.parent / "__tmp__"
        # srt_path.parent.mkdir(parents=True, exist_ok=True)
        # 目标输出路径
        if args.inplace:
            merge_path = out_path_merge_inplace(file_path)
        else:
            merge_path = out_path_merge_mirror(args.input, args.output, file_path)
        if args.inplace:
            srt_path  = out_path_srt_inplace(file_path)
        else:
            srt_path = out_path_srt_mirror(args.input, args.output, file_path)
        if args.inplace:
            txt_path = out_path_txt_inplace(file_path)
        else:
            txt_path = out_path_txt_mirror(args.input, args.output, file_path)
        if args.inplace:
            json_path = out_path_json_inplace(file_path)
        else:
            json_path = out_path_json_mirror(args.input, args.output, file_path)


        tmp_work = merge_path.parent / "__tmp__"
        merge_path.parent.mkdir(parents=True, exist_ok=True)

        # 跳过已存在（可通过 --overwrite 控制）
        if not args.overwrite and srt_path.exists() and txt_path.exists() and json_path.exists():
            return (srt_path, True, "skip-exist")

        # 准备音频
        prepared = ensure_wav(file_path, tmp_work)

        # 推理
        result = pipe.transcribe(prepared)

        # 写结果
        # write_srt(result.get("segments", []), srt_path)  # 这三行 与下面write_all_outputs 重复
        # write_txt(result.get("text", ""), txt_path)
        # write_json(result, json_path)
        # from segment_merger import write_merge_file     不再需要这个模块了
        # 统一写入所有输出格式（含 merge.txt）
        base_path = mirror_output_path(args.input, args.output, file_path, ".srt").with_suffix("")
        write_all_outputs(result, base_path)

        # 清理临时
        if prepared != file_path and prepared.name.endswith(".__tmp__.wav"):
            try:
                prepared.unlink(missing_ok=True)
                if tmp_work.exists():
                    shutil.rmtree(tmp_work, ignore_errors=True)
            except Exception:
                pass

        return (srt_path, True, "ok")
    except subprocess.CalledProcessError as e:
        return (file_path, False, f"ffmpeg-fail: {e}")
    except Exception as e:
        return (file_path, False, f"infer-fail: {e}")


def collect_files(input_dir: Path, recursive: bool) -> List[Path]:
    pattern = "**/*" if recursive else "*"
    files = []
    for p in input_dir.glob(pattern):
        if p.is_file() and p.suffix.lower() in SUPPORTED_ALL:
            files.append(p)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(r"C:\Users\Administrator\Downloads\input"))
    parser.add_argument("--output", type=Path, default=Path(r"C:\Users\Administrator\Downloads\output"))
    parser.add_argument("--model-dir", type=str, default=None, help="本地模型缓存/权重目录（离线）")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--no-diarize", action="store_true")
    parser.add_argument("--recursive", action="store_true", default=True, help="递归扫描子目录（默认开启）")
    parser.add_argument("--inplace", action="store_true", default=True, help="在源文件所在目录生成输出（默认开启）")
    parser.add_argument("--workers", type=int, default=1, help="并发数（GPU 建议 1）")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.input = args.input.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    files = collect_files(args.input, args.recursive)
    if not files:
        print(f"[WARN] 未找到可处理文件：{args.input}")
        return

    pipe = InferencePipeline(
        model_dir=args.model_dir,
        diarize=not args.no_diarize,
        device=args.device,
    )

    if args.workers <= 1:
        for f in files:
            path, ok, msg = process_one(f, args, pipe)
            print(f"[{'OK' if ok else 'ERR'}] {f} -> {path} ({msg})")
    else:
        with futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            tasks = {ex.submit(process_one, f, args, pipe): f for f in files}
            for t in futures.as_completed(tasks):
                path, ok, msg = t.result()
                src = tasks[t]
                print(f"[{'OK' if ok else 'ERR'}] {src} -> {path} ({msg})")


if __name__ == "__main__":
    main()
