"""Export to GGUF: convert an HF directory to GGUF with the official llama.cpp script, then
quantize it and do a test run (Chapter 20).

    uv run python -m zero.export.gguf --hf-dir out/tiny/hf_chat --out out/tiny/model-f16.gguf \\
        --quantize Q8_0 --run "<|im_start|>user\\n你好<|im_end|>\\n<|im_start|>assistant\\n"

Steps:

1. `ensure_llama_cpp()`: make a shallow clone of https://github.com/ggml-org/llama.cpp in a cache
   directory (default `~/.cache/zero/llama.cpp`; the environment variable `ZERO_LLAMA_CPP` can
   point to an existing repository).
2. `convert_hf_to_gguf()`: call the official `convert_hf_to_gguf.py` in the repository. It needs
   numpy / torch / transformers and the `gguf-py` folder of the repository (the script adds it to
   sys.path automatically; you do not need to install the `gguf` package).
   **The only change**: the official script identifies the pre-tokenizer from "a hash of the
   tokenizer output for a test text". The hash of our own tokenizer is not in its table, so the
   script reports "BPE pre-tokenizer was not recognized". Our pre-tokenization regex is exactly
   the same as Qwen2 (see PRETOKENIZE_REGEX in zero/tokenizer.py). Thus, when the script does not
   know the hash, we fall back to `"qwen2"`. This is a runtime patch before the script runs; it
   does not change the llama.cpp files. `tests/test_gguf.py` does a token-by-token parity check
   with `llama-tokenize`. It makes sure that llama.cpp tokenizes the same way as our tokenizer.
3. `build_llama_cpp()`: build the CPU versions of `llama-quantize`, `llama-simple`,
   `llama-completion`, and `llama-tokenize` with cmake (if cmake is missing:
   `apt-get install -y cmake`).
4. `quantize()`: `llama-quantize in.gguf out.gguf Q4_K_M` (or Q8_0 and others).
5. `run_llama()`: `llama-simple` or `llama-completion` generates some tokens. This shows that the
   model "runs on a laptop".

Note: k-quants such as Q4_K_M need tensor rows whose length is a multiple of 256. Most matrices of
the tiny model (dim=128) do not meet this condition, so llama-quantize automatically uses a
different quantization type for them. The main-line model (dim=1280) does not have this problem.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

LLAMA_CPP_URL = "https://github.com/ggml-org/llama.cpp"
DEFAULT_TARGETS = ("llama-quantize", "llama-simple", "llama-completion", "llama-tokenize")


def default_llama_cpp_dir() -> Path:
    env = os.environ.get("ZERO_LLAMA_CPP")
    if env:
        return Path(env)
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "zero" / "llama.cpp"


def ensure_llama_cpp(path: str | os.PathLike | None = None, ref: str | None = None) -> Path:
    """Make sure that a local llama.cpp repository exists (else make a shallow clone), and return its path."""
    d = Path(path) if path else default_llama_cpp_dir()
    if (d / "convert_hf_to_gguf.py").exists():
        return d
    d.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1"]
    if ref:
        cmd += ["--branch", ref]
    subprocess.run([*cmd, LLAMA_CPP_URL, str(d)], check=True)
    return d


def llama_cpp_commit(d: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# A patch that runs before the official script: if the pre-tokenizer is unknown, fall back to qwen2
# (our regex is the same as Qwen2)
_SHIM = r"""
import runpy, sys
llama_dir, pre = sys.argv[1], sys.argv[2]
sys.path.insert(0, llama_dir)
sys.path.insert(1, llama_dir + "/gguf-py")
from conversion import base
_orig = base.TextModel.get_vocab_base_pre
def _patched(self, tokenizer):
    try:
        return _orig(self, tokenizer)
    except NotImplementedError:
        print(f"[zero] pre-tokenizer not registered; use {pre!r} (the zero regex is the same as Qwen2)", file=sys.stderr)
        return pre
base.TextModel.get_vocab_base_pre = _patched
sys.argv = [llama_dir + "/convert_hf_to_gguf.py"] + sys.argv[3:]
runpy.run_path(llama_dir + "/convert_hf_to_gguf.py", run_name="__main__")
"""


def convert_hf_to_gguf(
    hf_dir: str | os.PathLike,
    out_file: str | os.PathLike,
    outtype: str = "f16",
    llama_cpp_dir: str | os.PathLike | None = None,
    pre_tokenizer: str = "qwen2",
    python: str | None = None,
) -> Path:
    """HF directory → GGUF file (outtype: f32 / f16 / bf16 / q8_0)."""
    d = ensure_llama_cpp(llama_cpp_dir)
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python or sys.executable,
        "-c",
        _SHIM,
        str(d),
        pre_tokenizer,
        str(hf_dir),
        "--outfile",
        str(out),
        "--outtype",
        outtype,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not out.exists():
        raise RuntimeError(
            f"convert_hf_to_gguf.py failed:\n{res.stdout[-2000:]}\n{res.stderr[-4000:]}"
        )
    return out


def build_llama_cpp(
    llama_cpp_dir: str | os.PathLike | None = None,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
    jobs: int = 2,
) -> Path:
    """Build some CPU programs with cmake and return the bin directory. If they are already built, return immediately."""
    d = ensure_llama_cpp(llama_cpp_dir)
    bin_dir = d / "build" / "bin"
    if all((bin_dir / t).exists() for t in targets):
        return bin_dir
    if shutil.which("cmake") is None:
        raise RuntimeError("cmake is missing: apt-get install -y cmake (or pip install cmake)")
    subprocess.run(
        [
            "cmake",
            "-B",
            str(d / "build"),
            "-S",
            str(d),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DLLAMA_CURL=OFF",
            "-DLLAMA_OPENSSL=OFF",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_SERVER=OFF",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["cmake", "--build", str(d / "build"), "-j", str(jobs), "--target", *targets],
        check=True,
        capture_output=True,
    )
    return bin_dir


def find_binary(name: str, llama_cpp_dir: str | os.PathLike | None = None) -> Path | None:
    d = Path(llama_cpp_dir) if llama_cpp_dir else default_llama_cpp_dir()
    p = d / "build" / "bin" / name
    if p.exists():
        return p
    w = shutil.which(name)
    return Path(w) if w else None


def quantize(
    gguf_in: str | os.PathLike,
    gguf_out: str | os.PathLike,
    qtype: str = "Q8_0",
    llama_cpp_dir: str | os.PathLike | None = None,
) -> Path:
    exe = find_binary("llama-quantize", llama_cpp_dir)
    if exe is None:
        raise FileNotFoundError("llama-quantize not found; run build_llama_cpp() first")
    res = subprocess.run(
        [str(exe), str(gguf_in), str(gguf_out), qtype], capture_output=True, text=True
    )
    if res.returncode != 0:
        raise RuntimeError(f"llama-quantize failed: {res.stderr[-3000:]}")
    return Path(gguf_out)


def run_llama(
    gguf: str | os.PathLike,
    prompt: str,
    n_tokens: int = 16,
    llama_cpp_dir: str | os.PathLike | None = None,
    threads: int = 1,
) -> str:
    """Generate n_tokens tokens with llama-simple (greedy) and return stdout (the prompt and the generated text)."""
    exe = find_binary("llama-simple", llama_cpp_dir)
    if exe is None:
        raise FileNotFoundError("llama-simple not found; run build_llama_cpp() first")
    res = subprocess.run(
        [str(exe), "-m", str(gguf), "-n", str(n_tokens), prompt],
        capture_output=True,
        timeout=300,
        env={**os.environ, "OMP_NUM_THREADS": str(threads)},
    )
    # llama-simple prints token by token, and a byte-level token can be half of a Chinese character.
    # Thus join all bytes first, then decode.
    if res.returncode != 0:
        raise RuntimeError(f"llama-simple failed: {res.stderr.decode('utf-8', 'replace')[-3000:]}")
    return res.stdout.decode("utf-8", errors="replace")


def llama_tokenize(
    gguf: str | os.PathLike, text: str, llama_cpp_dir: str | os.PathLike | None = None
) -> list[int]:
    """Encode with the llama.cpp tokenizer (no BOS) and return the token ids (for a parity check with our tokenizer)."""
    exe = find_binary("llama-tokenize", llama_cpp_dir)
    if exe is None:
        raise FileNotFoundError("llama-tokenize not found")
    res = subprocess.run(
        [str(exe), "-m", str(gguf), "-p", text, "--ids", "--no-bos", "--log-disable"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if res.returncode != 0:
        raise RuntimeError(f"llama-tokenize failed: {res.stderr[-2000:]}")
    line = [ln for ln in res.stdout.strip().splitlines() if ln.startswith("[")][-1]
    return [int(x) for x in line.strip("[]").split(",") if x.strip()]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="HF directory → GGUF (Chapter 20)")
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--out", required=True, help="output GGUF file")
    ap.add_argument("--outtype", default="f16", choices=["f32", "f16", "bf16", "q8_0"])
    ap.add_argument("--quantize", default="", help="then quantize to this type, for example Q4_K_M or Q8_0")
    ap.add_argument("--run", default="", help="after quantization, run this prompt with llama-simple")
    ap.add_argument("--llama-cpp", default=None, help="path of the llama.cpp repository (default: the cache directory)")
    args = ap.parse_args(argv)
    out = convert_hf_to_gguf(args.hf_dir, args.out, args.outtype, args.llama_cpp)
    print(f"GGUF: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    final = out
    if args.quantize or args.run:
        build_llama_cpp(args.llama_cpp)
    if args.quantize:
        final = quantize(
            out, out.with_name(out.stem + f"-{args.quantize}.gguf"), args.quantize, args.llama_cpp
        )
        print(f"Quantized: {final} ({final.stat().st_size / 1e6:.1f} MB)")
    if args.run:
        print(run_llama(final, args.run, llama_cpp_dir=args.llama_cpp))


if __name__ == "__main__":
    main()
