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

def format_hms(t: float) -> str:
    # 时间戳格式化为 00:00:00
    h, rem = divmod(int(t), 3600)
    m, rem = divmod(rem, 60)
    s = int(rem)
    return f"{h:02d}:{m:02d}:{s:02d}"

def write_merge_date_txt(segments: List[Dict[str, Any]], out_path: Path, min_chars: int = 200, max_chars: int = 250):
    def split_with_timestamps(text: str, start: float, end: float) -> List[Tuple[str, str, str]]:
        """
        将文本分割成 200–250 字左右的段落（按句号），并估算时间戳区间。
        返回 [(start_str, end_str, text), ...]
        """
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
            if len(cur) < min_chars or (len(cur) + len(s) <= max_chars):
                cur += s
            else:
                paras.append(cur)
                cur = s
        if cur:
            paras.append(cur)

        # 时间分段估算（线性均分）
        total = len(paras)
        t_start = float(start)
        t_end = float(end)
        t_step = (t_end - t_start) / max(total, 1)
        out = []
        for i, para in enumerate(paras):
            ts_start = t_start + i * t_step
            ts_end = ts_start + t_step
            out.append((format_hms(ts_start), format_hms(ts_end), para.strip()))
        return out

    lines = []
    last_spk = None
    block_text = ""
    block_start = None
    block_end = None

    def flush_block():
        nonlocal block_text, block_start, block_end, last_spk
        if not block_text or block_start is None or block_end is None:
            return
        chunks = split_with_timestamps(block_text, block_start, block_end)
        for i, (ts1, ts2, text) in enumerate(chunks):
            lines.append(f"{ts1}-{ts2}")
            # 第一段标注 speaker，后续分段仍保留 speaker
            lines.append(f"speaker {last_spk}")
            lines.append(text)
            lines.append("")  # 段落间空行
        block_text = ""
        block_start = None
        block_end = None

    for seg in segments:
        spk = seg.get("speaker", "UNK")
        text = seg.get("text", "").strip()
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))

        if last_spk is None:
            last_spk = spk
            block_start = start
            block_end = end
            block_text = text
            continue

        if spk != last_spk:
            flush_block()
            last_spk = spk
            block_start = start
            block_end = end
            block_text = text
        else:
            # 累计时间和文本
            block_text += text
            block_end = end

    flush_block()  # 最后一段

    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")



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
    write_merge_date_txt(segments, base_path.with_name(base_path.stem + "_merge.date.txt"))  # ✅ 新增这行


def process_one(file_path: Path, args, pipe: InferencePipeline) -> Tuple[Path, bool, str]:
    """
    返回：(srt_path, success, msg)
    """
    try:
        # 输出路径（同名不同后缀）
        srt_path = mirror_output_path(args.input, args.output, file_path, ".srt")
        txt_path = mirror_output_path(args.input, args.output, file_path, ".txt")
        json_path = mirror_output_path(args.input, args.output, file_path, ".json")
        tmp_work = srt_path.parent / "__tmp__"
        srt_path.parent.mkdir(parents=True, exist_ok=True)

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
    parser.add_argument("--recursive", action="store_true")
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
