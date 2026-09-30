"""导出 GGUF：用 llama.cpp 官方脚本把 HF 目录转成 GGUF，再量化、试跑（对应第 20 章）。

    uv run python -m zero.export.gguf --hf-dir out/tiny/hf_chat --out out/tiny/model-f16.gguf \\
        --quantize Q8_0 --run "<|im_start|>user\\n你好<|im_end|>\\n<|im_start|>assistant\\n"

步骤：

1. `ensure_llama_cpp()`：把 https://github.com/ggml-org/llama.cpp 浅克隆到缓存目录
   （默认 `~/.cache/zero/llama.cpp`，可用环境变量 `ZERO_LLAMA_CPP` 指定已有的仓库）；
2. `convert_hf_to_gguf()`：调用仓库里的官方 `convert_hf_to_gguf.py`。它依赖 numpy / torch /
   transformers，以及仓库自带的 `gguf-py`（脚本会自动把它加进 sys.path，不需要另装 `gguf` 包）。
   **唯一的改动**：官方脚本用"分词器对一段测试文本的编码结果的哈希"识别预切分规则，我们自训的分词器
   哈希不在它的表里，会报 "BPE pre-tokenizer was not recognized"。我们的预切分正则与 Qwen2 完全相同
   （见 zero/tokenizer.py 的 PRETOKENIZE_REGEX），所以在不认识时回退为 `"qwen2"`——这是在调用脚本前
   打的一个运行时补丁，不修改 llama.cpp 的文件。`tests/test_gguf.py` 用 `llama-tokenize` 逐 token 对拍，
   确认 llama.cpp 的分词与我们的分词器一致；
3. `build_llama_cpp()`：cmake 编译 CPU 版的 `llama-quantize`、`llama-simple`、`llama-completion`、
   `llama-tokenize`（没有 cmake 时 `apt-get install -y cmake`）；
4. `quantize()`：`llama-quantize in.gguf out.gguf Q4_K_M`（或 Q8_0 等）；
5. `run_llama()`：`llama-simple` 或 `llama-completion` 生成几个 token，证明"笔记本上能跑"。

注意：Q4_K_M 等 k-quant 要求张量的行长是 256 的倍数，tiny 模型（dim=128）的大部分矩阵不满足，
llama-quantize 会对这些张量自动回退到别的量化类型；主线模型（dim=1280）没有这个问题。
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
    """确保本地有 llama.cpp 仓库（没有就浅克隆），返回路径。"""
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


# 在官方脚本之前执行的补丁：预切分规则不认识时回退为 qwen2（我们的正则与 Qwen2 相同）
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
        print(f"[zero] 预切分规则未登记，按 {pre!r} 处理（zero 的正则与 Qwen2 相同）", file=sys.stderr)
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
    """HF 目录 → GGUF 文件（outtype：f32 / f16 / bf16 / q8_0）。"""
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
            f"convert_hf_to_gguf.py 失败：\n{res.stdout[-2000:]}\n{res.stderr[-4000:]}"
        )
    return out


def build_llama_cpp(
    llama_cpp_dir: str | os.PathLike | None = None,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
    jobs: int = 2,
) -> Path:
    """cmake 编译 CPU 版的若干个程序，返回 bin 目录。已经编译过的直接返回。"""
    d = ensure_llama_cpp(llama_cpp_dir)
    bin_dir = d / "build" / "bin"
    if all((bin_dir / t).exists() for t in targets):
        return bin_dir
    if shutil.which("cmake") is None:
        raise RuntimeError("没有 cmake：apt-get install -y cmake（或 pip install cmake）")
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
        raise FileNotFoundError("找不到 llama-quantize，先 build_llama_cpp()")
    res = subprocess.run(
        [str(exe), str(gguf_in), str(gguf_out), qtype], capture_output=True, text=True
    )
    if res.returncode != 0:
        raise RuntimeError(f"llama-quantize 失败：{res.stderr[-3000:]}")
    return Path(gguf_out)


def run_llama(
    gguf: str | os.PathLike,
    prompt: str,
    n_tokens: int = 16,
    llama_cpp_dir: str | os.PathLike | None = None,
    threads: int = 1,
) -> str:
    """用 llama-simple 生成 n_tokens 个 token（贪心），返回 stdout（含提示词和生成的文本）。"""
    exe = find_binary("llama-simple", llama_cpp_dir)
    if exe is None:
        raise FileNotFoundError("找不到 llama-simple，先 build_llama_cpp()")
    res = subprocess.run(
        [str(exe), "-m", str(gguf), "-n", str(n_tokens), prompt],
        capture_output=True,
        timeout=300,
        env={**os.environ, "OMP_NUM_THREADS": str(threads)},
    )
    # llama-simple 逐 token 打印，字节级 token 可能是半个汉字；整段字节拼起来再解码
    if res.returncode != 0:
        raise RuntimeError(f"llama-simple 失败：{res.stderr.decode('utf-8', 'replace')[-3000:]}")
    return res.stdout.decode("utf-8", errors="replace")


def llama_tokenize(
    gguf: str | os.PathLike, text: str, llama_cpp_dir: str | os.PathLike | None = None
) -> list[int]:
    """用 llama.cpp 的分词器编码（不加 BOS），返回 token id 列表（与我们的分词器对拍用）。"""
    exe = find_binary("llama-tokenize", llama_cpp_dir)
    if exe is None:
        raise FileNotFoundError("找不到 llama-tokenize")
    res = subprocess.run(
        [str(exe), "-m", str(gguf), "-p", text, "--ids", "--no-bos", "--log-disable"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if res.returncode != 0:
        raise RuntimeError(f"llama-tokenize 失败：{res.stderr[-2000:]}")
    line = [ln for ln in res.stdout.strip().splitlines() if ln.startswith("[")][-1]
    return [int(x) for x in line.strip("[]").split(",") if x.strip()]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="HF 目录 → GGUF（第 20 章）")
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--out", required=True, help="输出的 GGUF 文件")
    ap.add_argument("--outtype", default="f16", choices=["f32", "f16", "bf16", "q8_0"])
    ap.add_argument("--quantize", default="", help="再量化成这个类型，如 Q4_K_M、Q8_0")
    ap.add_argument("--run", default="", help="量化后用 llama-simple 跑这个提示词")
    ap.add_argument("--llama-cpp", default=None, help="llama.cpp 仓库路径（默认缓存目录）")
    args = ap.parse_args(argv)
    out = convert_hf_to_gguf(args.hf_dir, args.out, args.outtype, args.llama_cpp)
    print(f"GGUF: {out}（{out.stat().st_size / 1e6:.1f} MB）")
    final = out
    if args.quantize or args.run:
        build_llama_cpp(args.llama_cpp)
    if args.quantize:
        final = quantize(
            out, out.with_name(out.stem + f"-{args.quantize}.gguf"), args.quantize, args.llama_cpp
        )
        print(f"量化: {final}（{final.stat().st_size / 1e6:.1f} MB）")
    if args.run:
        print(run_llama(final, args.run, llama_cpp_dir=args.llama_cpp))


if __name__ == "__main__":
    main()
