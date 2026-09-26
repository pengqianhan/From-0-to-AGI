"""旁白合成：把每一镜的旁白逐句合成语音，记录每句的起止时间（用来生成字幕）。

后端（环境变量 VIDEO_TTS 选择）：
- sherpa（默认）：离线的 sherpa-onnx + MeloTTS 中英混读模型，MIT 许可，结果可复现，
  不依赖网络服务。模型不存在时自动从 GitHub Releases 下载。
- edge：edge-tts（微软在线语音），需要能访问 speech.platform.bing.com 的 WebSocket。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
import urllib.request
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .script import Shot

SAMPLE_RATE = 44100
SENTENCE_GAP = 0.4   # 句与句之间的停顿（秒）
SHOT_TAIL = 0.6      # 每镜旁白结束后的留白（秒）
DEFAULT_SPEED = float(os.environ.get("VIDEO_TTS_SPEED", "0.9"))  # 1.0 约每秒 6 个汉字，偏快
PEAK_TARGET = 0.7    # 约 -3 dBFS：每镜统一响度，同时给 AAC 编码留余量

_MELO_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-melo-tts-zh_en.tar.bz2"
)


@dataclass
class SentenceTiming:
    display: str
    start: float  # 相对本镜开始的秒数
    end: float


@dataclass
class ShotAudio:
    shot_id: str
    wav: str
    duration: float
    sentences: list[SentenceTiming]


def _model_dir() -> Path:
    d = Path(os.environ.get("VIDEO_TTS_MODEL_DIR", Path.home() / ".cache/tts"))
    target = d / "vits-melo-tts-zh_en"
    if not (target / "model.onnx").exists():
        d.mkdir(parents=True, exist_ok=True)
        archive = d / "vits-melo-tts-zh_en.tar.bz2"
        print(f"[tts] 下载 TTS 模型到 {archive} …")
        urllib.request.urlretrieve(_MELO_URL, archive)
        with tarfile.open(archive) as tf:
            tf.extractall(d)  # noqa: S202 - 固定来源的官方模型包
    return target


class _SherpaBackend:
    name = "sherpa-melo-zh_en"

    def __init__(self, speed: float = 1.0) -> None:
        import sherpa_onnx

        d = _model_dir()
        vits = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=str(d / "model.onnx"),
            lexicon=str(d / "lexicon.txt"),
            tokens=str(d / "tokens.txt"),
            dict_dir=str(d / "dict"),
        )
        fsts = ",".join(
            str(d / f) for f in ["date.fst", "number.fst", "phone.fst", "new_heteronym.fst"]
        )
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(vits=vits, num_threads=os.cpu_count() or 2),
            rule_fsts=fsts,
        )
        self.tts = sherpa_onnx.OfflineTts(cfg)
        self.speed = speed

    def synth(self, text: str) -> np.ndarray:
        audio = self.tts.generate(text, sid=0, speed=self.speed)
        x = np.asarray(audio.samples, dtype=np.float32)
        if audio.sample_rate != SAMPLE_RATE:
            x = _resample(x, audio.sample_rate, SAMPLE_RATE)
        return x


class _EdgeBackend:
    name = "edge-tts"

    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural", speed: float = 1.0) -> None:
        self.voice = voice
        self.speed = speed
        pct = int(round((speed - 1.0) * 100))
        self.rate = f"{pct:+d}%"

    def synth(self, text: str) -> np.ndarray:
        import asyncio

        import edge_tts

        with tempfile.TemporaryDirectory() as tmp:
            mp3 = Path(tmp) / "s.mp3"
            wav = Path(tmp) / "s.wav"

            async def _run() -> None:
                await edge_tts.Communicate(text, self.voice, rate=self.rate).save(str(mp3))

            asyncio.run(_run())
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3), "-ac", "1",
                 "-ar", str(SAMPLE_RATE), str(wav)],
                check=True,
            )
            return _read_wav(wav)


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    n_out = int(round(len(x) * sr_out / sr_in))
    return np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x).astype(np.float32)


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768.0


def _write_wav(path: Path, x: np.ndarray) -> None:
    pcm = (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def make_backend(name: str | None = None, speed: float | None = None):
    name = name or os.environ.get("VIDEO_TTS", "sherpa")
    speed = DEFAULT_SPEED if speed is None else speed
    if name == "sherpa":
        return _SherpaBackend(speed=speed)
    if name == "edge":
        return _EdgeBackend(speed=speed)
    raise ValueError(f"未知的 TTS 后端：{name}")


def synthesize_shots(shots: list[Shot], out_dir: Path, backend=None) -> list[ShotAudio]:
    """逐镜合成旁白；按文本哈希缓存，旁白没改就不重新合成。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = backend or make_backend()
    results: list[ShotAudio] = []
    for shot in shots:
        key = hashlib.sha1(
            (f"{backend.name}|{backend.speed}|{SENTENCE_GAP}|{SHOT_TAIL}|{PEAK_TARGET}|" + shot.narration).encode("utf-8")
        ).hexdigest()[:12]
        wav_path = out_dir / f"{shot.shot_id}_{key}.wav"
        meta_path = wav_path.with_suffix(".json")
        if wav_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            results.append(
                ShotAudio(shot.shot_id, str(wav_path), meta["duration"],
                          [SentenceTiming(**s) for s in meta["sentences"]])
            )
            continue
        pieces: list[np.ndarray] = []
        timings: list[SentenceTiming] = []
        t = 0.0
        gap = np.zeros(int(SENTENCE_GAP * SAMPLE_RATE), dtype=np.float32)
        for i, sent in enumerate(shot.sentences):
            x = backend.synth(sent.spoken)
            dur = len(x) / SAMPLE_RATE
            timings.append(SentenceTiming(sent.display, round(t, 3), round(t + dur, 3)))
            pieces.append(x)
            t += dur
            if i < len(shot.sentences) - 1:
                pieces.append(gap)
                t += SENTENCE_GAP
        pieces.append(np.zeros(int(SHOT_TAIL * SAMPLE_RATE), dtype=np.float32))
        audio = np.concatenate(pieces)
        peak = float(np.abs(audio).max()) if len(audio) else 0.0
        if peak > 0:
            audio = audio * (PEAK_TARGET / peak)
        _write_wav(wav_path, audio)
        duration = len(audio) / SAMPLE_RATE
        meta_path.write_text(
            json.dumps({"duration": duration, "sentences": [asdict(s) for s in timings]},
                       ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        results.append(ShotAudio(shot.shot_id, str(wav_path), duration, timings))
        print(f"[tts] {shot.shot_id}: {duration:.1f}s，{len(timings)} 句")
    return results
