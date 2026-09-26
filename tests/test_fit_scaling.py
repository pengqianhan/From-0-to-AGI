"""scaling law 拟合与预算规划（zero/tools/fit_scaling.py、plan_budget.py，第 12 章）的测试。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from zero.config import load_model_config
from zero.tools.estimate_cost import estimate_cost
from zero.tools.fit_scaling import (
    ChinchillaFit,
    Point,
    bootstrap_chinchilla,
    fit_chinchilla,
    fit_loss_to_score,
    fit_power_law,
    load_points,
    main,
    model_size,
)
from zero.tools.plan_budget import apply_candidate, plan, tokens_for_budget

TRUE = ChinchillaFit(E=1.82, A=482.0, B=2085.0, alpha=0.348, beta=0.366)  # Epoch 复现的量级


def _ladder(noise: float = 0.0, seed: int = 0) -> list[Point]:
    rng = np.random.default_rng(seed)
    pts = []
    for N in [2e7, 6e7, 1.5e8, 3e8]:
        for ratio in [5, 20, 80]:
            D = N * ratio
            L = float(TRUE.predict(N, D)) * (1 + rng.normal(0, noise))
            pts.append(Point(N, D, L, name=f"{N:.0e}x{ratio}"))
    return pts


def test_fit_recovers_known_parameters() -> None:
    fit = fit_chinchilla(_ladder())
    assert fit.E == pytest.approx(TRUE.E, abs=0.02)
    assert fit.alpha == pytest.approx(TRUE.alpha, abs=0.02)
    assert fit.beta == pytest.approx(TRUE.beta, abs=0.02)
    assert fit.rmse < 1e-3
    assert not fit.at_boundary
    # 外推 2 倍尺寸、长得多的训练：误差 < 0.3%
    N, D = 6e8, 4e11
    assert float(fit.predict(N, D)) == pytest.approx(float(TRUE.predict(N, D)), rel=3e-3)
    # 算力最优比例与真值一致
    n_opt, d_opt = fit.compute_optimal(1e21)
    t_n, t_d = TRUE.compute_optimal(1e21)
    assert d_opt / n_opt == pytest.approx(t_d / t_n, rel=0.1)


def test_bootstrap_interval_covers_truth_with_noise() -> None:
    pts = _ladder(noise=0.003, seed=1)
    fit = fit_chinchilla(pts)
    boot = bootstrap_chinchilla(pts, n=40, seed=0)
    N, D = 6e8, 4e11
    lo, hi = boot.interval(lambda f: float(f.predict(N, D)))
    assert lo < float(fit.predict(N, D)) < hi
    assert lo < float(TRUE.predict(N, D)) < hi


def test_tied_exponents_and_power_law() -> None:
    fit = fit_chinchilla(_ladder(), tie_exponents=True)
    assert fit.alpha == fit.beta
    C = np.array([1e18, 1e19, 1e20, 1e21, 1e22])
    L = 2.0 + 3.0 * (C / 1e20) ** -0.15
    pl = fit_power_law(C, L)
    assert pl.E == pytest.approx(2.0, abs=0.01)
    assert pl.gamma == pytest.approx(0.15, abs=0.005)
    assert float(pl.predict(1e24)) == pytest.approx(2.0 + 3.0 * 1e4**-0.15, rel=1e-3)


def test_loss_to_score_sigmoid() -> None:
    losses = np.linspace(3.5, 2.0, 12)
    scores = 0.25 + 0.75 / (1 + np.exp(-4.0 * (2.4 - losses)))
    sf = fit_loss_to_score(losses, scores, chance=0.25)
    assert sf.L0 == pytest.approx(2.4, abs=0.02)
    assert sf.k == pytest.approx(4.0, rel=0.05)


def test_cli_with_trainer_logs(tmp_path, capsys) -> None:
    """用训练器格式的 log.jsonl + 配置文件拟合，并读留出运行、写出 JSON。"""
    runs = []
    for i, (dim, ffn) in enumerate([(64, 192), (96, 256), (128, 384), (160, 448), (192, 512)]):
        cfg_path_i = tmp_path / f"m{i}.toml"
        cfg_path_i.write_text(
            f"[model]\nvocab_size = 256\ndim = {dim}\nn_layers = 2\nn_heads = 4\nn_kv_heads = 2\n"
            f"ffn_dim = {ffn}\nmax_seq_len = 64\n[data]\nseq_len = 64\n"
        )
        N = model_size(load_model_config(cfg_path_i))
        for ratio in [10, 40]:
            run = tmp_path / f"r{i}_{ratio}"
            run.mkdir()
            D = N * ratio
            recs = [
                {"step": 1, "loss": 5.0, "tokens": 1000, "val_loss": None},
                {"step": 2, "loss": 3.0, "tokens": int(D), "val_loss": float(TRUE.predict(N, D))},
            ]
            (run / "log.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
            runs.append(f"{run}:{cfg_path_i}")
    out = tmp_path / "fit.json"
    main(
        [
            *sum((["--run", r] for r in runs[:-2]), []),
            "--holdout",
            runs[-1],
            "--target-N",
            "600M",
            "--target-D",
            "400B",
            "--bootstrap",
            "10",
            "--out",
            str(out),
        ]
    )
    res = json.loads(out.read_text())
    ho = res["holdout"][0]
    assert ho["pred"] == pytest.approx(ho["loss"], rel=0.01)
    assert "外推" in capsys.readouterr().out
    pts_file = tmp_path / "pts.jsonl"
    pts_file.write_text(
        "\n".join(json.dumps({"N": p.N, "D": p.D, "loss": p.loss}) for p in _ladder())
    )
    assert len(load_points(pts_file)) == 12


def test_plan_matches_estimate_cost() -> None:
    cfg = load_model_config("configs/main/pretrain.toml")
    tokens = tokens_for_budget(cfg, 5000.0, 4096, mfu=0.4)
    # 反过来用 estimate_cost 算这么多 token 的费用，必须正好是预算
    assert estimate_cost(cfg, tokens, 4096, mfu=0.4).cost_usd == pytest.approx(5000.0, rel=1e-9)
    row = plan(cfg, 5000.0, 4096, mfu=0.4)
    assert row.cost_usd == pytest.approx(5000.0, rel=1e-9)
    assert row.tokens_per_param == pytest.approx(tokens / row.params_total)
    # 与 RUNBOOK 的数字一致：500B token 约 $6.1K → $5K 约 410B
    assert 405e9 < tokens < 415e9
    # 提速 1.25 倍 ≡ MFU 从 0.4 变 0.5
    assert tokens_for_budget(cfg, 5000.0, 4096, mfu=0.4, speedup=1.25) == pytest.approx(
        tokens_for_budget(cfg, 5000.0, 4096, mfu=0.5)
    )
    name, small = apply_candidate(cfg, "L24:n_layers=24")
    assert name == "L24" and small.n_layers == 24
    assert tokens_for_budget(small, 5000.0, 4096) > tokens
