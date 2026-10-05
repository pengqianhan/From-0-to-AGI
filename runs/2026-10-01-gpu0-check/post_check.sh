#!/usr/bin/env bash
# 阶段 6 第 9、10 项的补充 + Muon 训练（单卡 GPU0），在 smoke_cuda.py 跑完之后执行：
#   flock <gpu0.lock> bash runs/2026-10-01-gpu0-check/post_check.sh
# 1. torchrun（nproc=1）入口的 SFT：从 smoke_cuda 的中期训练 checkpoint 出发，20 步，BF16 autocast；
# 2. HF 导出 + transformers 在 GPU 上对拍（export_check.py）：smoke_cuda 的 GRPO 模型、第 2 项的 l20m 模型；
# 3. optim.name=muon 的 l20m 训练 100 步（MuonAdamW 的 CUDA BF16 NS5 路径）。
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1
S=out/gpu0-check/smoke_cuda
O=out/gpu0-check

echo "== 1. torchrun SFT（nproc=1）"
rm -rf $O/sft_torchrun
uv run torchrun --standalone --nproc_per_node=1 -m zero.post.sft --config configs/tiny/sft.toml \
  --set train.device=cuda --set train.dtype=auto --set train.max_steps=20 --set train.eval_every=10 \
  --set train.micro_batch_size=8 --set model.max_seq_len=512 --set data.seq_len=512 \
  --set model.rope_scaling.factor=4.0 \
  --set train.out_dir="\"$O/sft_torchrun\"" --set train.init_from="\"$S/midtrain/ckpt\"" \
  --set data.tokenizer="\"$S/tokenizer.json\"" \
  --set sft.train_jsonl="\"$S/sft/train.jsonl\"" --set sft.val_jsonl="\"$S/sft/val.jsonl\"" \
  --set sft.shard_dir="\"$O/sft_torchrun/data\"" > $O/sft_torchrun.log 2>&1
grep -E "模型参数|step +(1|10|20)/|加载|Error" $O/sft_torchrun.log

echo "== 2. HF 导出对拍"
uv run python runs/2026-10-01-gpu0-check/export_check.py $S/grpo/ckpt $O/export_grpo 2>&1 | grep -v Warning
uv run python runs/2026-10-01-gpu0-check/export_check.py $O/pretrain_l20m/ckpt $O/export_l20m 2>&1 | grep -v Warning

echo "== 3. Muon（l20m，100 步）"
bash runs/2026-10-01-gpu0-check/train_run.sh pretrain_l20m_muon runs/2026-10-01-gpu0-check/configs/l20m_tiny.toml \
  --set optim.name=muon --set train.max_steps=100 --set schedule.warmup_steps=20 \
  --set checkpoint.every=100 --set train.eval_every=100 --set logging.every=10
