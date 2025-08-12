import os
from pathlib import Path
from typing import Dict, Any
from funasr import AutoModel
import psutil
import torch

class InferencePipeline:
    def __init__(self, model_dir: str, device: str = "cuda", diarize: bool = True):
        base_path = Path(model_dir).resolve()
        self.device = device
        self.diarize = diarize

        # 每个模型路径 
        self.asr_model_path = base_path / "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        self.vad_model_path = base_path / "speech_fsmn_vad_zh-cn-16k-common-pytorch"
        self.punc_model_path = base_path / "punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
        self.spk_model_path = base_path / "speech_campplus_sv_zh-cn_16k-common"
        
        # 版本信息
        self.model_revision = "v2.0.4"
        self.hotword_file = "./hotwords.txt"
        self.ngpu = 1 if device == "cuda" and torch.cuda.is_available() else 0
        self.ncpu = psutil.cpu_count()

        # 构造 AutoModel 参数
        auto_args = {
            "model": str(self.asr_model_path),
            "model_revision": self.model_revision,
            "vad_model": str(self.vad_model_path),
            "vad_model_revision": self.model_revision,
            "punc_model": str(self.punc_model_path),
            "punc_model_revision": self.model_revision,
            "ngpu": self.ngpu,
            "ncpu": self.ncpu,
            "device": self.device,
            "disable_pbar": True,
            "disable_log": True,
            "disable_update": True
        }

        # 仅当启用说话人识别时添加 spk_model
        if self.diarize:
            auto_args.update({
                "spk_model": str(self.spk_model_path),
                "spk_model_revision": self.model_revision
            })

        self.model = AutoModel(**auto_args)




        # 热词
        self.hotwords = ""
        if os.path.exists(self.hotword_file):
            with open(self.hotword_file, encoding="utf-8") as f:
                self.hotwords = " ".join([line.strip() for line in f])

    def transcribe(self, audio_path: str | Path) -> Dict[str, Any]:
        audio_path = str(audio_path)

        # 使用 ffmpeg 抽音轨为 bytes
        import ffmpeg
        audio_bytes, _ = (
            ffmpeg.input(audio_path, threads=0)
            .output("pipe:", format="wav", acodec="pcm_s16le", ac=1, ar=16000)
            .run(capture_stdout=True, capture_stderr=True)
        )

        res = self.model.generate(
            input=audio_bytes,
            batch_size_s=300,
            is_final=True,
            sentence_timestamp=True,
            hotword=self.hotwords
        )

        # 抽取结构化信息
        rec_result = res[0]
        sentences = []
        for s in rec_result.get("sentence_info", []):
            sentences.append({
                "start": s["start"] / 1000,  # ms to sec
                "end": s["end"] / 1000,
                "text": s["text"],
                "speaker": s["spk"]
            })

        return {
            "text": rec_result.get("text", ""),
            "segments": sentences,
            "language": "zh",
            "duration": rec_result.get("time_stamp", {}).get("duration", 0.0)
        }
