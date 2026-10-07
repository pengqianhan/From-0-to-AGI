"""Narration synthesis: synthesize the narration of each shot sentence by sentence.

It records the start and end time of each sentence (for the subtitles).

Backends (select one with the environment variable VIDEO_TTS):
- sherpa (default): offline sherpa-onnx with the MeloTTS model for mixed Chinese and English.
  MIT license, reproducible results, no network service. If the model is not on the disk,
  the script downloads it from GitHub Releases.
- edge: edge-tts (Microsoft online voices). It needs access to the WebSocket of
  speech.platform.bing.com.
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
SENTENCE_GAP = 0.4   # pause between two sentences (s)
SHOT_TAIL = 0.6      # silence after the narration of each shot (s)
DEFAULT_SPEED = float(os.environ.get("VIDEO_TTS_SPEED", "0.9"))  # 1.0 is about 6 Chinese characters/s, a bit fast
PEAK_TARGET = 0.7    # about -3 dBFS: the same loudness in each shot, with headroom for the AAC encoder

_MELO_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-melo-tts-zh_en.tar.bz2"
)


@dataclass
class SentenceTiming:
    display: str
    start: float  # seconds from the start of the shot
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
        print(f"[tts] Downloading the TTS model to {archive} …")
        urllib.request.urlretrieve(_MELO_URL, archive)
        with tarfile.open(archive) as tf:
            tf.extractall(d)  # noqa: S202 - official model archive from a fixed source
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
        # The default uses all cores. On a shared server, set VIDEO_TTS_THREADS to limit the threads.
        threads = int(os.environ.get("VIDEO_TTS_THREADS", "0")) or os.cpu_count() or 2
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(vits=vits, num_threads=threads),
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
    raise ValueError(f"Unknown TTS backend: {name}")


def synthesize_shots(shots: list[Shot], out_dir: Path, backend=None) -> list[ShotAudio]:
    """Synthesize the narration shot by shot.

    The cache key is a hash of the text. If the narration did not change, the shot is not
    synthesized again.
    """
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
        print(f"[tts] {shot.shot_id}: {duration:.1f}s, {len(timings)} sentences")
    return results
