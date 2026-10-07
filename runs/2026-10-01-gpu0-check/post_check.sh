#!/usr/bin/env bash
# Additions to Stage 6, items 9 and 10, + Muon training (1 GPU, GPU0). Run this script after smoke_cuda.py:
#   flock <gpu0.lock> bash runs/2026-10-01-gpu0-check/post_check.sh
# 1. SFT through the torchrun entry point (nproc=1): start from the mid-training checkpoint of smoke_cuda,
#    20 steps, BF16 autocast;
# 2. HF export + parity check with transformers on the GPU (export_check.py): the GRPO model of smoke_cuda
#    and the l20m model of item 2;
# 3. l20m training with optim.name=muon for 100 steps (the CUDA BF16 NS5 path of MuonAdamW).
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1
S=out/gpu0-check/smoke_cuda
O=out/gpu0-check

echo "== 1. torchrun SFT (nproc=1)"
rm -rf $O/sft_torchrun
uv run torchrun --standalone --nproc_per_node=1 -m zero.post.sft --config configs/tiny/sft.toml \
  --set train.device=cuda --set train.dtype=auto --set train.max_steps=20 --set train.eval_every=10 \
  --set train.micro_batch_size=8 --set model.max_seq_len=512 --set data.seq_len=512 \
  --set model.rope_scaling.factor=4.0 \
  --set train.out_dir="\"$O/sft_torchrun\"" --set train.init_from="\"$S/midtrain/ckpt\"" \
  --set data.tokenizer="\"$S/tokenizer.json\"" \
  --set sft.train_jsonl="\"$S/sft/train.jsonl\"" --set sft.val_jsonl="\"$S/sft/val.jsonl\"" \
  --set sft.shard_dir="\"$O/sft_torchrun/data\"" > $O/sft_torchrun.log 2>&1
grep -E "Model parameters|step +(1|10|20)/|Loaded|Error" $O/sft_torchrun.log

echo "== 2. HF export parity check"
uv run python runs/2026-10-01-gpu0-check/export_check.py $S/grpo/ckpt $O/export_grpo 2>&1 | grep -v Warning
uv run python runs/2026-10-01-gpu0-check/export_check.py $O/pretrain_l20m/ckpt $O/export_l20m 2>&1 | grep -v Warning

echo "== 3. Muon (l20m, 100 steps)"
bash runs/2026-10-01-gpu0-check/train_run.sh pretrain_l20m_muon runs/2026-10-01-gpu0-check/configs/l20m_tiny.toml \
  --set optim.name=muon --set train.max_steps=100 --set schedule.warmup_steps=20 \
  --set checkpoint.every=100 --set train.eval_every=100 --set logging.every=10
