"""阶梯实验的固定验证集：按训练配比从各来源的验证分片里拼出正好 1024 个长 2048 的片段。

    uv run python runs/ladder-3090/build_val.py      # → data/ladder3090/ladder_val.bin + ladder_val.json

为什么不直接用流水线写出的 <来源>_val_*.bin：训练器评估时不打乱、只取前 eval_batches × micro_batch 个片段，
按文件名排序会先读到同一个来源；而且 micro batch 随搜索的配置变化，不同配置看到的验证数据就不一样。
这里把片段数定成 1024（能被任何 2 的幂次 micro batch 整除），配置里 eval_batches 设得足够大，
每次评估都完整跑一遍同一份数据，所有配置的 val_bpb 才能直接比较。
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from zero.data.shard import TOKEN_DTYPE, read_shard  # noqa: E402

DATA = REPO / "data" / "ladder3090"
N_CHUNKS, SEQ = 1024, 2048


def main() -> None:
    cfg = tomllib.loads((REPO / "configs" / "ladder3090" / "base.toml").read_text("utf-8"))
    weights = {s["name"]: s["weight"] for s in cfg["data"]["sources"]}
    # 按权重分配片段数（最大余数法，保证总数正好 1024）
    raw = {k: w / sum(weights.values()) * N_CHUNKS for k, w in weights.items()}
    alloc = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - alloc[k], reverse=True)[: N_CHUNKS - sum(alloc.values())]:
        alloc[k] += 1
    parts, record = [], {}
    for name, n in alloc.items():
        files = sorted(DATA.glob(f"{name}_val_*.bin"))
        stream = np.concatenate([np.asarray(read_shard(f)) for f in files]) if files else np.empty(0, TOKEN_DTYPE)
        need = n * SEQ
        if len(stream) < need + 1:
            raise SystemExit(f"{name} 的验证分片只有 {len(stream):,} 个 token，需要 {need + 1:,}")
        parts.append(stream[:need])
        record[name] = {"chunks": n, "tokens": need, "from": [f.name for f in files]}
    parts.append(stream[need : need + 1])  # 最后一个片段还要往后看一个 token（片段长 seq_len + 1）
    tokens = np.concatenate(parts).astype(TOKEN_DTYPE)
    assert (len(tokens) - 1) // SEQ == N_CHUNKS
    tokens.tofile(DATA / "ladder_val.bin")
    (DATA / "ladder_val.json").write_text(json.dumps(
        {"chunks": N_CHUNKS, "seq_len": SEQ, "tokens": int(len(tokens)), "sources": record},
        ensure_ascii=False, indent=2), "utf-8")
    for k, v in record.items():
        print(f"{k:18s} {v['chunks']:5d} 个片段  {v['tokens']:>10,} token")
    print(f"共 {len(tokens):,} token → {DATA / 'ladder_val.bin'}")


if __name__ == "__main__":
    main()
