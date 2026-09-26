"""第 18 章 · 极简代码 1：Bradley–Terry 奖励模型

一条偏好数据只说"a 比 b 好"，不给分数。Bradley–Terry 模型假设每个回答有一个看不见的分数 r，
    P(a ≻ b) = σ(r(a) − r(b))
训练奖励模型就是最大化这个概率的对数——本质上是第 5 章的二分类交叉熵（标签永远是"chosen 赢"）。

玩具设定：每个回答用两个特征描述——质量 q 和长度 ℓ（单位：百 token）。
标注员按 r_label = 1.5·q + 0.4·ℓ 做判断（有噪声地：按 σ 的概率挑），也就是说**标注员略微偏爱长回答**。
我们只看得到"谁赢了"，看看奖励模型能不能把这两个系数学回来。

运行：uv run python chapters/18-preference-alignment/code/01_bradley_terry.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

torch.set_num_threads(1)  # 构建机多任务共享 CPU（本机可删）

W_LABEL = torch.tensor([1.5, 0.4])  # 标注员心里的打分：1.5·质量 + 0.4·长度


def make_answers(n: int, g: torch.Generator) -> torch.Tensor:
    """n 个回答的特征 (n, 2)：质量 q ~ N(0,1)，长度 ℓ ~ U(0.2, 4)（百 token）。"""
    q = torch.randn(n, generator=g)
    length = 0.2 + 3.8 * torch.rand(n, generator=g)
    return torch.stack([q, length], dim=1)


def make_pairs(n_pairs: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (chosen 特征, rejected 特征)，各 (n_pairs, 2)。标注员按 Bradley–Terry 概率挑赢家。"""
    g = torch.Generator().manual_seed(seed)
    a, b = make_answers(n_pairs, g), make_answers(n_pairs, g)
    p_a_wins = torch.sigmoid(a @ W_LABEL - b @ W_LABEL)  # P(a ≻ b) = σ(r(a) − r(b))
    a_wins = torch.rand(n_pairs, generator=g) < p_a_wins
    chosen = torch.where(a_wins[:, None], a, b)
    rejected = torch.where(a_wins[:, None], b, a)
    return chosen, rejected


class LinearRM(torch.nn.Module):
    """最简单的奖励模型：r(x) = w·x + c。真实的奖励模型是"LLM + 一个输出标量的头"，损失完全一样。"""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(2, 1)
        torch.nn.init.zeros_(self.lin.weight)  # 从"什么都不懂"出发：所有回答同分，损失 = log 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).squeeze(-1)


def bt_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    """Bradley–Terry 负对数似然：−log σ(r_w − r_l)。"""
    return -F.logsigmoid(r_chosen - r_rejected).mean()


def train_rm(steps: int = 600, lr: float = 0.1, seed: int = 0) -> tuple[LinearRM, list[dict]]:
    torch.manual_seed(seed)
    cw, cl = make_pairs(4000, seed=1)
    rm = LinearRM()
    c0 = float(rm.lin.bias.detach())
    opt = torch.optim.Adam(rm.parameters(), lr=lr)
    hist = []
    for step in range(steps + 1):
        loss = bt_loss(rm(cw), rm(cl))
        if step in (0, 10, 50, 200, steps):
            w = rm.lin.weight.detach()[0]
            hist.append({"step": step, "loss": float(loss.detach()), "w_q": float(w[0]), "w_len": float(w[1])})
        opt.zero_grad()
        loss.backward()
        opt.step()
    rm.bias_moved = abs(float(rm.lin.bias.detach()) - c0)  # type: ignore[attr-defined]
    return rm, hist


def accuracy(r_fn, cw: torch.Tensor, cl: torch.Tensor) -> float:  # noqa: ANN001
    return float((r_fn(cw) > r_fn(cl)).float().mean())


def main() -> None:
    print("① σ 把'分数差'变成'赢的概率'：P(a ≻ b) = σ(r(a) − r(b))")
    print("   r(a) − r(b):", "  ".join(f"{d:+.0f}" for d in (-4, -2, -1, 0, 1, 2, 4)))
    print("   P(a ≻ b)   :", "  ".join(f"{torch.sigmoid(torch.tensor(float(d))):.2f}" for d in (-4, -2, -1, 0, 1, 2, 4)))
    print("   注意：只有'差'进公式——所有回答的分数同时加 100，概率不变。")

    rm, hist = train_rm()
    print("\n② 在 4000 对偏好数据上训练线性奖励模型 r(x) = w_q·质量 + w_len·长度 + c")
    print(f"   {'步数':>4} | {'损失':>6} | {'w_q':>6} | {'w_len':>6}")
    for h in hist:
        print(f"   {h['step']:>4} | {h['loss']:.4f} | {h['w_q']:6.3f} | {h['w_len']:6.3f}")
    print(f"   标注员真实的系数：w_q = {W_LABEL[0]:.1f}，w_len = {W_LABEL[1]:.1f}")
    print(f"   偏置 c 训练前后变化：{rm.bias_moved:.1e}（梯度恒为 0：常数在'差'里被消掉了）")

    cw, cl = make_pairs(4000, seed=2)  # 留出集
    with torch.no_grad():
        acc_rm = accuracy(rm, cw, cl)
        acc_oracle = accuracy(lambda x: x @ W_LABEL, cw, cl)
        acc_quality = accuracy(lambda x: x[:, 0], cw, cl)
    print("\n③ 留出集上'奖励模型给 chosen 打分更高'的比例")
    print(f"   学到的奖励模型         : {acc_rm:.3f}")
    print(f"   标注员自己的打分（上限）: {acc_oracle:.3f}  ← 标注有噪声，上限不是 1")
    print(f"   只看质量 q             : {acc_quality:.3f}")

    # ④ 它就是二分类交叉熵
    with torch.no_grad():
        d = rm(cw) - rm(cl)
        ours = bt_loss(rm(cw), rm(cl))
        bce = F.binary_cross_entropy_with_logits(d, torch.ones_like(d))
    print("\n④ Bradley–Terry 损失 = 以 r_w − r_l 为 logit、标签恒为 1 的二分类交叉熵（第 5 章）")
    print(f"   −log σ(r_w − r_l) = {float(ours):.6f}   BCEWithLogits = {float(bce):.6f}")
    print(f"\n小结：奖励模型把'长度'也当成了优点（w_len ≈ {hist[-1]['w_len']:.2f}）。"
          "它忠实地学会了标注员的偏好——包括偏见。下一步（02）看看策略拼命追这个分数会怎样。")


if __name__ == "__main__":
    main()
