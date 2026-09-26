"""第 14 章 · 极简代码 7：断点续训 —— 恢复"全部"状态，loss 才能逐位一致

预训练要跑几天到几周，机器一定会出事：抢占、坏卡、断电、loss 爆了要回滚。所以要定期存 checkpoint，
崩了从最近的 checkpoint 接着训。"接着训"的标准是：续训后的每一步，和从没中断过**逐位相同**。

要做到这一点，光存模型权重远远不够。一次训练的全部状态是：
  1. 模型参数
  2. 优化器状态（AdamW 的 m、v，以及步数 t —— 偏差修正要用）
  3. 学习率调度器的位置
  4. 数据读到哪了（下一批是哪几条）
  5. 随机数发生器的状态（dropout、数据打乱等用到的所有 RNG）
  6. 步数等计数器

这个脚本：先不中断训练 30 步；再训练到第 17 步"崩溃"（最近的 checkpoint 在第 15 步），
在全新的对象里从 checkpoint 恢复、训到第 30 步，逐步比较 loss；最后依次"忘掉"某一项状态，看差多少。

运行：uv run python chapters/14-pretraining-engineering/code/07_resume.py   （几秒）
"""

import io
import math

import torch
from torch import nn

STEPS, SAVE_AT, CRASH_AT = 30, 15, 17


class Run:
    """一次训练的全部可变状态都挂在这里。"""

    def __init__(self):
        torch.manual_seed(0)
        self.model = nn.Sequential(nn.Linear(32, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 8))
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=3e-3, weight_decay=0.1)
        warmup = 5
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * s / STEPS)))
        self.data_gen = torch.Generator().manual_seed(1234)    # 数据加载器自己的随机数（抽哪几条）
        g = torch.Generator().manual_seed(42)
        self.X = torch.randn(4096, 32, generator=g)
        self.Y = (self.X[:, :8] * 2 + 0.3 * torch.randn(4096, 8, generator=g)).argmax(1)
        self.step = 0

    def train_step(self) -> float:
        idx = torch.randint(0, len(self.X), (64,), generator=self.data_gen)
        loss = nn.functional.cross_entropy(self.model(self.X[idx]), self.Y[idx])  # dropout 用全局 RNG
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.sched.step()
        self.step += 1
        return loss.item()

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optim": self.opt.state_dict(),
            "sched": self.sched.state_dict(),
            "data_rng": self.data_gen.get_state(),
            "torch_rng": torch.get_rng_state(),
            "step": self.step,
        }

    def load_state_dict(self, sd: dict, skip: str = "") -> None:
        self.model.load_state_dict(sd["model"])
        if skip != "optim":
            self.opt.load_state_dict(sd["optim"])
        if skip != "sched":
            self.sched.load_state_dict(sd["sched"])
        if skip != "data_rng":
            self.data_gen.set_state(sd["data_rng"])
        if skip != "torch_rng":
            torch.set_rng_state(sd["torch_rng"])
        self.step = sd["step"]


def resume(ckpt: bytes, skip: str = "") -> list[float]:
    torch.manual_seed(999)                  # 模拟"新进程"：全局随机数状态被打乱
    run = Run()                             # 全新的模型、优化器、调度器、数据加载器
    run.load_state_dict(torch.load(io.BytesIO(ckpt), weights_only=False), skip)
    return [run.train_step() for _ in range(run.step, STEPS)]


def main():
    torch.set_num_threads(1)
    ref_run = Run()
    ref = [ref_run.train_step() for _ in range(STEPS)]

    run = Run()
    ckpt = None
    for _ in range(CRASH_AT):
        run.train_step()
        if run.step == SAVE_AT:
            buf = io.BytesIO()
            torch.save(run.state_dict(), buf)   # 生产代码里写到磁盘（先写临时目录再原子改名）
            ckpt = buf.getvalue()
    print(f"训练到第 {CRASH_AT} 步时崩溃；最近的 checkpoint 在第 {SAVE_AT} 步（{len(ckpt):,} 字节）")
    resumed = resume(ckpt)

    print(f"\n① 恢复全部状态：第 {SAVE_AT + 1}–{STEPS} 步的 loss")
    print(f"  {'步':>3} {'不中断':>10} {'续训':>10}  相同？")
    for s in (SAVE_AT + 1, SAVE_AT + 2, CRASH_AT + 1, STEPS):
        a, b = ref[s - 1], resumed[s - SAVE_AT - 1]
        print(f"  {s:>3} {a:>10.6f} {b:>10.6f}  {'逐位相同' if a == b else '不同'}")
    same = all(a == b for a, b in zip(ref[SAVE_AT:], resumed))
    print(f"  全部 {STEPS - SAVE_AT} 步逐位相同：{same}")

    print("\n② 少恢复一项，会怎样（续训 15 步里 loss 与不中断的最大差）")
    labels = {
        "optim": "不恢复优化器状态（m、v、t 从零开始）",
        "sched": "不恢复学习率调度（从 warmup 重新开始）",
        "data_rng": "不恢复数据位置（重新抽到开头那几批）",
        "torch_rng": "不恢复全局 RNG（dropout 掩码不同）",
    }
    for key, label in labels.items():
        diff = max(abs(a - b) for a, b in zip(ref[SAVE_AT:], resume(ckpt, skip=key)))
        print(f"  {label:<24} 最大差 {diff:.2e}")


if __name__ == "__main__":
    main()
