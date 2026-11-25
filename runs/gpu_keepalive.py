"""Keep the GPU reservation alive: the server cancels a reservation after 60 minutes without GPU use.

    setsid nohup env UV_NO_SYNC=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python runs/gpu_keepalive.py --loop \
        >> out/gpu_keepalive.log 2>&1 < /dev/null &

One round: if no compute process is on the card, run matrix multiplications for --seconds (about 1 GB).
With --loop, repeat every --every minutes. If a training run is on the card, the round does nothing.
Stop it with: kill $(cat out/gpu_keepalive.pid)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]


def busy() -> bool:
    out = subprocess.run(
        ["nvidia-smi", "-i", GPU, "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    ).stdout
    return any(line.strip() and int(line) != os.getpid() for line in out.splitlines())


def burn(seconds: float) -> None:
    import torch

    x = torch.randn(8192, 8192, device="cuda")
    end = time.time() + seconds
    while time.time() < end:
        (x @ x).sum().item()
    del x
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--every", type=float, default=50, help="minutes between rounds (with --loop)")
    ap.add_argument("--loop", action="store_true")
    args = ap.parse_args()
    if args.loop:
        Path("out/gpu_keepalive.pid").write_text(str(os.getpid()))
    while True:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        if busy():
            print(f"{stamp} GPU {GPU} is in use, skip", flush=True)
        else:
            burn(args.seconds)
            print(f"{stamp} GPU {GPU} was idle, ran {args.seconds:.0f} s", flush=True)
        if not args.loop:
            return
        time.sleep(args.every * 60)


if __name__ == "__main__":
    main()
