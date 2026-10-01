"""阶梯实验的总调度（runs/ladder-3090/README.md 第 5 节）：按阶段依次执行，脱离会话运行约 6 天。

    setsid nohup uv run python runs/ladder-3090/orchestrate.py > out/ladder3090/orchestrate.log 2>&1 &

每一步都是幂等的（sweep.py 跳过已经训到的配置、采样只追加），中断后直接重跑本脚本就会从断点接着走。
阶段之间的决定按事先写好的规则自动做，记在 runs/ladder-3090/results/decisions.json：

- 逐轮淘汰：e5m 的 128 组全部训满后模拟"如果按 2/8、4/8 处的成绩淘汰会选出谁"。选中的配置比真正最好的
  差不超过 0.5%、并且全局前 5 名至少留下 3 个，e11m 起才采用；否则改为全部训满、配置数减半（算力不变）。
- 收窄的搜索空间：e24m、e44m 一半配置从完整空间抽、一半按更小几档的结果收窄（sweep.py narrow_ranges）；
  e83m 全部从收窄的空间抽。
- e44m 训完先拟合一次 L(N, D)，在 e83m 开训之前把对它的预测存下来（results/fit_before_e83m.json），
  e83m 训完再算外推误差——预测在先、检验在后。

GPU：e5m 只用 GPU0；e11m、e24m 用 GPU0 和 GPU2；e44m、e83m 用三张卡（GPU1 功耗上限 200W，慢一些，
只影响时间、不影响 loss）。
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = REPO / "runs" / "ladder-3090"
RES = HERE / "results"
GPUS = {"small": ["0"], "mid": ["0", "2"], "all": ["0", "1", "2"]}
SH_MAX_REGRET, SH_MIN_TOP5 = 0.005, 3


def log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def run(*args: object, fatal: bool = True) -> bool:
    cmd = [sys.executable, *map(str, args)]
    log("$ " + " ".join(cmd[1:]))
    ok = subprocess.run(cmd, cwd=REPO).returncode == 0
    if not ok:
        if fatal:
            raise SystemExit(f"失败，停止：{' '.join(cmd[1:])}")
        log("（分析步骤失败，不影响训练，继续）")
    return ok


def sweep(*a: object) -> bool:
    return run(HERE / "sweep.py", *a)


def analyze(*a: object) -> bool:
    return run(HERE / "analyze.py", *a, fatal=False)


def decide(key: str, value: object, why: str) -> None:
    path = RES / "decisions.json"
    d = json.loads(path.read_text()) if path.exists() else {}
    if key not in d:  # 决定只做一次；重跑时沿用第一次的
        d[key] = {"value": value, "why": why, "time": time.strftime("%F %T")}
        RES.mkdir(exist_ok=True)
        path.write_text(json.dumps(d, indent=2, ensure_ascii=False))
    log(f"决定 {key} = {d[key]['value']}：{d[key]['why']}")


def finish_scale(scale: str, gpus: list[str], top_branch: int = 8) -> None:
    sweep("branch", "--scale", scale, "--top", top_branch, "--gpus", *gpus)
    sweep("collect", "--scale", scale)
    analyze("frontier", "--scale", scale)
    analyze("nql", "--scale", scale)
    analyze("sensitivity", "--scale", scale)
    sweep("prune", "--scale", scale, "--keep-top", 8)


def run_scale(scale: str, n_full: int, n_narrow: int, narrow_from: list[str], gpus: list[str], use_sh: bool) -> None:
    log(f"===== {scale} =====")
    if not use_sh:
        n_full, n_narrow = math.ceil(n_full / 2), math.ceil(n_narrow / 2)
    sweep("sample", "--scale", scale, "--n", n_full)
    if n_narrow:
        sweep("sample", "--scale", scale, "--n", n_full + n_narrow, "--space", "narrow", "--from", *narrow_from)
    if use_sh:
        sweep("run", "--scale", scale, "--until", "2/8", "--gpus", *gpus)
        sweep("run", "--scale", scale, "--until", "4/8", "--top", "0.5", "--rank-at", "2/8", "--gpus", *gpus)
        sweep("run", "--scale", scale, "--until", "1", "--top", "0.25", "--rank-at", "4/8", "--gpus", *gpus)
    else:
        sweep("run", "--scale", scale, "--until", "1", "--gpus", *gpus)
    finish_scale(scale, gpus)


def main() -> None:
    log("总调度开始")
    # 第 2 阶段：最小档，128 组全部训满（也用来验证逐轮淘汰）
    log("===== e5m =====")
    sweep("sample", "--scale", "e5m", "--n", 128)
    sweep("run", "--scale", "e5m", "--until", "1", "--gpus", *GPUS["small"])
    finish_scale("e5m", GPUS["small"])
    sh = json.loads((RES / "e5m_frontier.json").read_text())["successive_halving"]
    use_sh = sh["regret_rel"] <= SH_MAX_REGRET and sh["global_top5_survived"] >= SH_MIN_TOP5
    decide("successive_halving", use_sh,
           f"e5m 模拟：选中的配置比真正最好的差 {sh['regret_rel']:.3%}（阈值 {SH_MAX_REGRET:.1%}），"
           f"全局前 5 名留下 {sh['global_top5_survived']} 个（阈值 {SH_MIN_TOP5}）")
    use_sh = json.loads((RES / "decisions.json").read_text())["successive_halving"]["value"]

    # 第 3 阶段：放大
    run_scale("e11m", 96, 0, [], GPUS["mid"], use_sh)
    run_scale("e24m", 24, 24, ["e5m", "e11m"], GPUS["mid"], use_sh)

    # 第 4 阶段：e44m，然后先拟合、存下对 e83m 的预测，再训 e83m 检验
    run_scale("e44m", 12, 12, ["e5m", "e11m", "e24m"], GPUS["all"], use_sh)
    if analyze("fit", "--fit", "e5m", "e11m", "e24m", "e44m") and not (RES / "fit_before_e83m.json").exists():
        shutil.copy(RES / "scaling_law.json", RES / "fit_before_e83m.json")
        shutil.copy(HERE / "figures" / "scaling_law.png", HERE / "figures" / "scaling_law_before_e83m.png")
    log("===== e83m（留出检验）=====")
    sweep("sample", "--scale", "e83m", "--n", 6, "--space", "narrow", "--from", "e5m", "e11m", "e24m", "e44m")
    sweep("run", "--scale", "e83m", "--until", "1", "--gpus", *GPUS["all"])
    finish_scale("e83m", GPUS["all"], top_branch=3)

    # 第 5 阶段：最终拟合（留出 e83m）
    analyze("fit", "--fit", "e5m", "e11m", "e24m", "e44m", "--holdout", "e83m")
    log("总调度完成")


if __name__ == "__main__":
    main()
