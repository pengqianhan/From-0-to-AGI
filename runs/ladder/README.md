# 阶梯实验（scaling ladder）协议 —— 第二步阶段 8、闸门 1 用

> 对应第 12 章（`chapters/12-scaling-laws/`）。本文件写于第一步：**下面的实验都还没有在 GPU 上跑过**，
> 费用是 `zero/tools/estimate_cost.py` 按 MFU 0.3 的估算（小模型 MFU 通常低于主线的 0.4），阶段 6 实测后更新。
> CPU 上的同一套流程见第 12 章 `code/03_lr_sweep.py`、`code/04_mini_ladder.py`（极小配置演示）。

## 1. 目的

用 4 个小模型回答闸门 1 的三个问题（GOAL.md 3.4）：

1. 按当前配方，主线 Base（`configs/main/pretrain.toml`，非 embedding 605.6M）在预算内的 token 数下，**验证 loss 能到多少**（附 95% 区间）；
2. 这个 loss 对应的**基准分数**大约是多少（两步法：loss → 分数）；
3. 学习率等超参按什么规则从小模型**迁移**到主线。

原则（第 12 章正文有来龙去脉）：

- **先调参再拟合**：每个尺寸的学习率单独扫描。没调好的小模型会让 scaling law 失真（Lourie et al. 2026，arXiv:2608.11859）。
- **配方固定**：数据配比、分词器（与主线相同的词表）、序列长度、优化器、调度、初始化，阶梯与主线完全一致；阶梯里改了任何一项，拟合就要重做（Delphi 第一次失败就是配方在长训练下不成立）。
- **覆盖主线所在的区域**：主线是约 600 token/参数的"过训练"区，阶梯必须有 token/参数远大于 20 的点，不能只跑 Chinchilla 最优点。
- **留出检验**：最大的尺寸不参与拟合，只用来检验外推误差（Delphi 的做法）。

## 2. 尺寸与 token 预算

共用 `configs/ladder/base.toml`：词表 65,536、序列 2,048、每步 262,144 token（16 × 8 卡 × 2048）、WSD 调度（最后 20% 线性衰减到 0）。

| 配置 | 总参数 | 非 embedding | token 预算（token/总参数） | 总步数（取整到 1000） | 用途 |
|---|---:|---:|---|---|---|
| `l20m` | 23.1M | 6.3M | 20× / 80× / 320× | 2,000 / 7,000 / 28,000 | 拟合 |
| `l60m` | 71.3M | 37.8M | 20× / 80× / 320× | 5,000 / 22,000 / 87,000 | 拟合 |
| `l150m` | 160.5M | 110.1M | 20× / 80× / 320× | 12,000 / 49,000 / 196,000 | 拟合 |
| `l300m` | 318.8M | 251.7M | 20× / 80× | 24,000 / 97,000 | **留出检验** |

每个尺寸只训练**一条主干**，用 WSD 分叉得到多个预算（第 12 章"一次训练得到多个点"）：

- 主干：`max_steps = 最大预算的步数`，训练到它的 80%（稳定段结束）；
- 预算 k 的分叉点 = `0.8 × S_k`。把主干在分叉点的 checkpoint 复制到新目录，用 `max_steps = S_k` 续训：
  前 80% 的学习率与主干完全相同（WSD 稳定段是常数），只多出最后 20% 的衰减；
- 成本 ≈ `0.8 × S_max + 0.2 × ΣS_k`，而不是 `ΣS_k`。

估算（MFU 0.3，$2.5/卡时，含分叉）：l20m ≈ $3、l60m ≈ $27、l150m ≈ $160、l300m ≈ $150，一轮合计约 **$340**；
加上学习率扫描（见 3）与 2 个种子的重复（见 5），阶梯总计约 **$450–550**，在 GOAL.md 3.4 给第 12–13 章的 $1,200 之内。
**l150m、l300m 的单次运行超过 $100，开跑前按 RUNBOOK 报批。**

## 3. 学习率扫描（先做）

- 在 20× 预算上，每个尺寸扫 5 个峰值学习率（相邻差 2 倍），以配置文件里的值为中心；**最优值落在网格边上就向外再扩一格**，直到最优在内部。
- l20m、l60m 各 5 个、l150m 3 个（围绕由小尺寸外推出来的值），l300m **不扫**——用外推值，这本身就是检验。
- 拟合 η\*(N) = c · N^(−k)（对数坐标下一条直线），外推到主线。Delphi 还在学习率里乘了训练长度修正 (T₀/T)^0.3：
  如果 80×、320× 预算的分叉点用 20× 上调出的学习率明显吃亏（在 l20m 上多跑两个学习率验证），就采用这个修正。
- 其他超参（β₁=0.9、β₂=0.95、权重衰减 0.1、warmup 200 步、梯度裁剪 1.0、batch 262K token）先固定。batch 与学习率的联合缩放（DeepSeek LLM 的 B_opt、η_opt 随算力的幂律）需要更多运行，本预算不做，如实记录为已知局限。

```bash
# 例：l60m 在 lr=1e-3 上的 20× 运行（其他学习率只改 --set optim.lr 和 out_dir）
uv run torchrun --standalone --nproc_per_node=8 -m zero.train.pretrain --config configs/ladder/l60m.toml \
  --set train.max_steps=5000 --set optim.lr=1e-3 --set train.out_dir=out/ladder/lr/l60m_1e-3
```

## 4. 主干与分叉的命令

```bash
S=l150m; LR=<扫描结果>
# 主干：训练到最大预算的 80%，每 800 步存一个 checkpoint，全部保留（分叉要用）
uv run torchrun --standalone --nproc_per_node=8 -m zero.train.pretrain --config configs/ladder/$S.toml \
  --set optim.lr=$LR --set train.max_steps=196000 --set checkpoint.every=800 --set checkpoint.keep_last=0 \
  --set train.out_dir=out/ladder/$S/trunk
# 分叉：预算 49,000 步 → 从 step 39,200 接一段衰减
mkdir -p out/ladder/$S/b49000/ckpt
cp -r out/ladder/$S/trunk/ckpt/step_00039200 out/ladder/$S/b49000/ckpt/
echo step_00039200 > out/ladder/$S/b49000/ckpt/latest
uv run torchrun --standalone --nproc_per_node=8 -m zero.train.pretrain --config configs/ladder/$S.toml \
  --set optim.lr=$LR --set train.max_steps=49000 --set train.out_dir=out/ladder/$S/b49000
```

- 主干本身在 `max_steps=196000` 时正好是最大预算的那一支（它训练到最后就包含了衰减段）。
- 所有步数取整到 1000 的倍数，分叉点就是 800 的倍数，与 `checkpoint.every=800` 对齐。
- 续训时 `LRScheduler` 按新的 `max_steps` 重建，分叉点之前的学习率与主干一致；数据加载器状态从 checkpoint 恢复，所以分叉看到的是主干"接下来"的数据。

## 5. 要记录什么

每个运行目录（`out/ladder/...`）由训练器自动写出 `log.jsonl`（`loss`、`val_loss`、`tokens`、`lr`、`grad_norm`、`tok_per_s`、`mfu`）。另外人工记录进 `runs/<日期>-ladder/README.md`：

- 每个 (尺寸, 预算) 的**最终** `val_loss`（衰减结束后那一次评估；中间点不能当作不同预算的结果）；
- 扫描表（尺寸 × 学习率 → val_loss）与选中的 η\*；
- 实测 `tok_per_s`、`mfu`（用于修正主线预算）；
- 至少在 l60m 的 20× 点上跑 2 个种子，记录种子间差异（Delphi：种子差约 0.1%，远小于外推区间）；
- 每个 checkpoint 上跑一遍预注册之外的**开发集**少样本评测（`zero.eval.harness`，不能用预注册的测试基准调参），同时记录"软指标"：正确选项的对数概率、参考答案的 bits-per-byte；
- loss spike、发散、重跑，全部如实记下。

## 6. 拟合与外推

```bash
uv run python -m zero.tools.fit_scaling \
  $(for d in out/ladder/l20m/* out/ladder/l60m/* out/ladder/l150m/*; do echo --run $d; done) \
  $(for d in out/ladder/l300m/*; do echo --holdout $d; done) \
  --target-config configs/main/pretrain.toml --target-D 400B --target-D 500B \
  --bootstrap 500 --out runs/ladder/fit.json
uv run python -m zero.tools.plan_budget --config configs/main/pretrain.toml --mfu <实测> --fit runs/ladder/fit.json
```

通过标准（写进闸门 1 报告）：

- 拟合点的最大相对误差 < 1%；
- l300m 留出点落在外推 95% 区间内，且误差 < 1%。超出时**不硬凑**：先查配方在长训练下是否失效（Delphi 第一次就是学习率没随训练长度调整），修配方、重跑阶梯；
- 外推到主线的区间宽度写清楚。主线相对 l150m 是 N 方向约 5.5 倍、D 方向约 8 倍的外推，区间会明显变宽。

## 7. 与第 12 章极小配置演示的对应

| 第二步（本协议） | 第 12 章 CPU 演示 |
|---|---|
| 4 个尺寸 × 3 个预算（WSD 分叉） | `code/04_mini_ladder.py`：4 个尺寸 × 3 个预算 |
| 每尺寸学习率扫描，最优在边上就外扩 | `code/03_lr_sweep.py` |
| η\*(N) 幂律外推给留出尺寸 | 同上，留出尺寸 s5 用外推学习率 |
| `zero.tools.fit_scaling`（非负最小二乘 + 分组 bootstrap） | `04_mini_ladder.py` 的 `fit_lnd` / `bootstrap` |
| l300m 留出检验 | s5（阶梯最大的 2.3 倍）留出检验 |
