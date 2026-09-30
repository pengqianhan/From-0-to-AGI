"""CUDA 对拍：zero 里标了"尚未在 GPU 上验证"的小模块，在 CUDA（FP32 / BF16 autocast）上能不能跑、
与 CPU FP32 的结果差多少（单张 RTX 3090）。

    CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/parity_cuda.py

内容：
1. 主线 Transformer 与第五部分的实验模块（MoE、MLA、滑动窗口、混合线性注意力 ×2、MTP）：
   同一份权重、同一批 token，CPU FP32 / CUDA FP32 / CUDA BF16 autocast 各做一次前向 + 反向，
   报告输出与梯度的相对误差 ‖a−b‖/‖b‖（以 CPU FP32 为基准）；
2. 缓存生成：各模块在 CUDA 上"带缓存"与"每步重算"的贪心输出是否逐字相同；推测解码 / MTP 自推测
   在 CUDA 上贪心时是否与目标模型贪心逐字相同；
3. 激活检查点在 CUDA 上梯度是否与不开时一致；
4. fused AdamW（trainer.py 在 CUDA 上默认打开）与 foreach AdamW / CPU AdamW 的参数差；
5. Muon：CUDA BF16 的 NS5 与 CPU FP32 的差距、MuonAdamW 在 CUDA 与 CPU 上走几步后的参数差；
6. checkpoint.py 里 CUDA 随机数状态的保存与恢复；
7. 几个模块在 CUDA 上的吞吐（只是"跑得动"的参考数字，不是性能验证）。
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from zero.arch import linear_attention as la  # noqa: E402
from zero.arch import mla as mla_mod  # noqa: E402
from zero.arch import sliding_window as sw  # noqa: E402
from zero.arch.moe import MoEConfig, moe_transformer  # noqa: E402
from zero.arch.mtp import MTPTransformer, mtp_loss, mtp_speculative_generate  # noqa: E402
from zero.arch.speculative import speculative_generate  # noqa: E402
from zero.config import ModelConfig  # noqa: E402
from zero.generate import generate  # noqa: E402
from zero.model import Transformer  # noqa: E402
from zero.train.checkpoint import rng_state, set_rng_state  # noqa: E402

CUDA = torch.device("cuda", 0)
RESULTS: dict = {}


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().double().cpu(), b.detach().double().cpu()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def grads(m: torch.nn.Module) -> torch.Tensor:
    return torch.cat(
        [
            (p.grad if p.grad is not None else torch.zeros_like(p))
            .detach()
            .double()
            .cpu()
            .flatten()
            for p in m.parameters()
        ]
    )


def base_cfg(**kw) -> ModelConfig:
    d = dict(
        vocab_size=512,
        dim=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        head_dim=32,
        ffn_dim=384,
        max_seq_len=256,
        init_std=0.1,
    )
    d.update(kw)
    return ModelConfig(**d)


# ---------------------------------------------------------------------------
# 1. 前向 + 反向对拍
# ---------------------------------------------------------------------------


def fwd_bwd(model, x, loss_fn, out_fn, device, bf16):
    model = model.to(device)
    model.zero_grad(set_to_none=True)
    x = x.to(device)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda")
    with ctx:
        out = out_fn(model, x)
        loss = loss_fn(model, x)
    loss.backward()
    return out.float().cpu(), float(loss), grads(model)


def parity(name: str, build, loss_fn, out_fn, B: int = 4, T: int = 128, vocab: int = 512) -> dict:
    torch.manual_seed(0)
    ref_model = build().float().train()
    x = torch.randint(0, vocab, (B, T + 1), generator=torch.Generator().manual_seed(1))
    runs = {}
    for tag, dev, bf16 in (
        ("cpu_fp32", torch.device("cpu"), False),
        ("cuda_fp32", CUDA, False),
        ("cuda_bf16", CUDA, True),
    ):
        m = copy.deepcopy(ref_model)
        try:
            runs[tag] = fwd_bwd(m, x, loss_fn, out_fn, dev, bf16)
        except Exception as e:  # noqa: BLE001 - 记录失败而不是中断整个脚本
            runs[tag] = e
    row: dict = {"module": name}
    base = runs["cpu_fp32"]
    for tag in ("cuda_fp32", "cuda_bf16"):
        r = runs[tag]
        if isinstance(r, Exception):
            row[tag] = {"ok": False, "error": f"{type(r).__name__}: {str(r).splitlines()[0][:160]}"}
            continue
        row[tag] = {
            "ok": True,
            "out_rel_err": f"{rel(r[0], base[0]):.2e}",
            "loss": f"{r[1]:.5f} vs {base[1]:.5f}",
            "grad_rel_err": f"{rel(r[2], base[2]):.2e}",
        }
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


def lm_loss(m, x):
    return m.loss(x[:, :-1], x[:, 1:])


def lm_out(m, x):
    return m(x[:, :-1])


def section_parity() -> list:
    rows = []
    # 主线 Transformer（对照组：说明 BF16 本身带来的误差量级）
    rows.append(
        parity("Transformer（主线，GQA）", lambda: Transformer(base_cfg()), lm_loss, lm_out)
    )

    # MoE：4 个路由专家 top-2 + 1 个共享专家；第 0 层稠密
    def build_moe():
        mc = base_cfg()
        cfg = MoEConfig.from_model_config(
            mc, n_experts=8, top_k=2, expert_dim=96, n_shared_experts=1, aux_loss_coef=0.01
        )
        return moe_transformer(mc, cfg, first_dense=1)

    rows.append(parity("MoE（8 选 2 + 共享专家，逐专家循环）", build_moe, lm_loss, lm_out))

    # MLA（训练时不吸收）
    def build_mla(q_lora=None):
        mc = base_cfg(n_kv_heads=4)
        cfg = mla_mod.MLAConfig.from_model_config(mc, kv_lora_rank=64, q_lora_rank=q_lora)
        return mla_mod.mla_transformer(mc, cfg)

    rows.append(parity("MLA", build_mla, lm_loss, lm_out))
    rows.append(parity("MLA（q_lora_rank=64）", lambda: build_mla(64), lm_loss, lm_out))

    # 滑动窗口：局部-全局 1:1，窗口 32
    def build_sw():
        m = Transformer(base_cfg())
        sw.convert_to_sliding_window(m, sw.make_layer_types(4, global_every=2), window=32)
        return m

    rows.append(parity("滑动窗口（窗口 32，局部:全局 = 1:1）", build_sw, lm_loss, lm_out))

    # 混合线性注意力（3:1）
    def build_hybrid(kind):
        cfg = la.HybridConfig(
            vocab_size=512,
            dim=128,
            n_layers=4,
            n_heads=4,
            n_kv_heads=2,
            ffn_dim=384,
            max_seq_len=256,
            init_std=0.1,
            full_attention_interval=4,
            linear_kind=kind,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_chunk_size=32,
        )
        return la.HybridTransformer(cfg)

    rows.append(
        parity(
            "Gated DeltaNet 混合（3:1）", lambda: build_hybrid("gated_deltanet"), lm_loss, lm_out
        )
    )
    rows.append(parity("线性注意力混合（3:1）", lambda: build_hybrid("linear"), lm_loss, lm_out))

    # MTP（深度 1）
    rows.append(
        parity(
            "MTP（深度 1）",
            lambda: MTPTransformer(base_cfg(), n_mtp=1),
            lambda m, x: mtp_loss(m, x[:, :-1], x[:, 1:])[0],
            lambda m, x: m(x[:, :-1])[0],
        )
    )
    return rows


# ---------------------------------------------------------------------------
# 2. 缓存生成 / 推测解码（CUDA 上）
# ---------------------------------------------------------------------------


def section_generation() -> dict:
    out: dict = {}
    prompt = [3, 17, 42, 5, 8, 8, 30, 11, 2, 99, 7, 64]

    def check(name, fn):
        try:
            same, info = fn()
            out[name] = {"ok": True, "identical": same, **info}
        except Exception as e:  # noqa: BLE001
            out[name] = {
                "ok": False,
                "error": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}",
            }
        print(name, json.dumps(out[name], ensure_ascii=False), flush=True)

    torch.manual_seed(0)
    base = Transformer(base_cfg(init_std=0.2)).to(CUDA).eval()

    def main_kv():
        a = generate(base, prompt, 60, temperature=0.0, use_cache=True)
        b = generate(base, prompt, 60, temperature=0.0, use_cache=False)
        return a == b, {"n": len(a)}

    check("主线 KV cache：带缓存 = 每步重算", main_kv)

    def mla_gen():
        mc = base_cfg(n_kv_heads=4, init_std=0.2)
        cfg = mla_mod.MLAConfig.from_model_config(mc, kv_lora_rank=64)
        torch.manual_seed(1)
        m = mla_mod.mla_transformer(mc, cfg).to(CUDA)
        a = mla_mod.generate_greedy(m, prompt, 60, cfg, use_cache=True)
        b = mla_mod.generate_greedy(m, prompt, 60, cfg, use_cache=False)
        return a == b, {"n": len(a)}

    check("MLA：吸收 + 潜向量缓存 = 每步重算", mla_gen)

    def sw_gen():
        torch.manual_seed(2)
        m = Transformer(base_cfg(init_std=0.2)).to(CUDA)
        sw.convert_to_sliding_window(m, sw.make_layer_types(4, global_every=2), window=16)
        cache = sw.SlidingWindowKVCache.from_model(m, batch_size=1, max_seq_len=128)
        a = sw.generate_greedy(m, prompt, 80, cache=cache)
        b = sw.generate_greedy(m, prompt, 80, cache=None)
        return a == b, {"n": len(a), "cache_device": str(cache.k[0].device)}

    check("滑动窗口：环形缓存 = 每步重算", sw_gen)

    for kind in ("gated_deltanet", "linear"):

        def hy_gen(kind=kind):
            torch.manual_seed(3)
            cfg = la.HybridConfig(
                vocab_size=512,
                dim=128,
                n_layers=4,
                n_heads=4,
                n_kv_heads=2,
                ffn_dim=384,
                max_seq_len=256,
                init_std=0.2,
                full_attention_interval=2,
                linear_kind=kind,
                linear_num_key_heads=2,
                linear_num_value_heads=4,
                linear_key_head_dim=32,
                linear_value_head_dim=32,
                linear_chunk_size=16,
            )
            m = la.HybridTransformer(cfg).to(CUDA).eval()
            a = la.generate_greedy(m, prompt, 60, use_cache=True)
            b = la.generate_greedy(m, prompt, 60, use_cache=False)
            return a == b, {"n": len(a)}

        check(f"混合 {kind}：递推状态 = 每步重算", hy_gen)

    def spec():
        torch.manual_seed(4)
        target = Transformer(base_cfg(init_std=0.2)).to(CUDA).eval()
        draft = Transformer(base_cfg(init_std=0.2, n_layers=1, dim=64, head_dim=16)).to(CUDA).eval()
        ref = generate(target, prompt, 60, temperature=0.0)
        res = speculative_generate(target, draft, prompt, 60, k=4, temperature=0.0)
        res_self = speculative_generate(target, target, prompt, 60, k=4, temperature=0.0)
        sampled = speculative_generate(target, draft, prompt, 60, k=4, temperature=0.8, seed=0)
        return res.tokens == ref and res_self.tokens == ref, {
            "rounds": res.rounds,
            "accepted": res.accepted,
            "self_draft_accept_rate": round(res_self.accepted / max(res_self.rounds * 4, 1), 3),
            "sampled_len": len(sampled.tokens),
        }

    check("推测解码：贪心 = 目标模型贪心（草稿 1 层 / 草稿 = 目标）", spec)

    def mtp_gen():
        torch.manual_seed(5)
        m = MTPTransformer(base_cfg(init_std=0.2), n_mtp=1).to(CUDA).eval()
        ref = generate(m.model, prompt, 60, temperature=0.0)
        res = mtp_speculative_generate(m, prompt, 60, temperature=0.0)
        s = mtp_speculative_generate(m, prompt, 60, temperature=1.0, top_p=0.95, seed=3)
        return res.tokens == ref, {
            "rounds": res.rounds,
            "accepted": res.accepted,
            "sampled_len": len(s.tokens),
        }

    check("MTP 自推测：贪心 = 主模型贪心", mtp_gen)
    return out


# ---------------------------------------------------------------------------
# 3–6. 激活检查点、fused AdamW、Muon、CUDA 随机数状态
# ---------------------------------------------------------------------------


def section_training_bits() -> dict:
    out: dict = {}
    # 3. 激活检查点
    torch.manual_seed(0)
    m = Transformer(base_cfg()).to(CUDA).train()
    x = torch.randint(0, 512, (4, 129), device=CUDA)
    res = {}
    for bf16 in (False, True):
        gs = []
        for ckpt in (False, True):
            m.activation_checkpointing = ckpt
            m.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                loss = m.loss(x[:, :-1], x[:, 1:])
            loss.backward()
            gs.append(grads(m))
        res["bf16" if bf16 else "fp32"] = f"{rel(gs[1], gs[0]):.2e}"
    out["activation_checkpointing_grad_rel_diff"] = res
    print("激活检查点（CUDA）开/关梯度相对差", res, flush=True)

    # 4. fused AdamW vs foreach AdamW vs CPU AdamW：同一串梯度，走 20 步
    torch.manual_seed(0)
    p0 = torch.randn(1024, 1024)
    gseq = [torch.randn(1024, 1024) * 1e-2 for _ in range(20)]
    finals = {}
    for tag, dev, kw in (
        ("cpu", "cpu", {}),
        ("cuda_foreach", CUDA, {"foreach": True}),
        ("cuda_fused", CUDA, {"fused": True}),
    ):
        p = torch.nn.Parameter(p0.clone().to(dev))
        opt = torch.optim.AdamW([p], lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1, **kw)
        for g in gseq:
            p.grad = g.to(dev)
            opt.step()
        finals[tag] = p.detach().cpu()
    out["fused_adamw"] = {
        "fused_vs_foreach_maxabs": f"{(finals['cuda_fused'] - finals['cuda_foreach']).abs().max():.2e}",
        "fused_vs_cpu_maxabs": f"{(finals['cuda_fused'] - finals['cpu']).abs().max():.2e}",
        "param_change_maxabs": f"{(finals['cpu'] - p0).abs().max():.2e}",
    }
    print("fused AdamW", out["fused_adamw"], flush=True)

    # 5. Muon：NS5 在 CUDA BF16 vs CPU FP32
    from zero.train.muon import build_muon_optimizer, zeropower_via_newtonschulz5

    ns = []
    for shape in ((1280, 1280), (2048, 1280), (1280, 3584), (3584, 1280)):
        torch.manual_seed(0)
        G = torch.randn(*shape)
        ref = zeropower_via_newtonschulz5(G)  # CPU：默认 FP32
        g_cuda = G.to(CUDA)
        cu = zeropower_via_newtonschulz5(g_cuda)  # CUDA：默认 BF16
        cu32 = zeropower_via_newtonschulz5(g_cuda, dtype=torch.float32)
        sv = torch.linalg.svdvals(cu.float()).cpu()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            zeropower_via_newtonschulz5(g_cuda)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 10 * 1e3
        ns.append(
            {
                "shape": list(shape),
                "cuda_dtype": str(cu.dtype),
                "bf16_vs_cpu_fp32_rel": f"{rel(cu, ref):.2e}",
                "cuda_fp32_vs_cpu_fp32_rel": f"{rel(cu32, ref):.2e}",
                "bf16_singular_values": [round(float(sv.min()), 3), round(float(sv.max()), 3)],
                "bf16_ms": round(ms, 2),
            }
        )
    out["muon_ns5"] = ns
    for r in ns:
        print("Muon NS5", r, flush=True)

    # MuonAdamW（build_muon_optimizer，与 trainer 相同）在 CUDA 与 CPU 上各走 5 步，比较参数的总更新量
    from zero.config import OptimConfig

    finals = {}
    torch.manual_seed(0)
    init_model = Transformer(base_cfg())
    init = torch.cat([p.detach().double().flatten() for p in init_model.parameters()])
    xx = torch.randint(0, 512, (4, 129), generator=torch.Generator().manual_seed(7))
    for dev in (torch.device("cpu"), CUDA):
        m = copy.deepcopy(init_model).to(dev)
        opt = build_muon_optimizer(m, OptimConfig(name="muon", lr=0.02, weight_decay=0.1), dev)
        xd = xx.to(dev)
        for _ in range(5):
            opt.zero_grad(set_to_none=True)
            m.loss(xd[:, :-1], xd[:, 1:]).backward()
            opt.step()
        finals[dev.type] = torch.cat([p.detach().double().cpu().flatten() for p in m.parameters()])
    d_cuda, d_cpu = finals["cuda"] - init, finals["cpu"] - init
    out["muon_optimizer_5_steps"] = {
        "update_rel_diff_cuda_vs_cpu": f"{rel(d_cuda, d_cpu):.2e}",
        "update_norm_cpu": f"{float(d_cpu.norm()):.3f}",
    }
    print("MuonAdamW 5 步", out["muon_optimizer_5_steps"], flush=True)

    # 6. CUDA 随机数状态：rng_state() / set_rng_state() 往返
    torch.cuda.manual_seed_all(123)
    _ = torch.rand(10, device=CUDA)
    st = rng_state()
    a = torch.rand(1000, device=CUDA)
    b_cpu = torch.rand(10)
    torch.rand(12345, device=CUDA)  # 把状态搅乱
    set_rng_state(st)
    a2 = torch.rand(1000, device=CUDA)
    b2 = torch.rand(10)
    out["cuda_rng_roundtrip"] = {
        "has_cuda_key": "cuda" in st,
        "n_cuda_states": len(st.get("cuda", [])),
        "cuda_identical": bool(torch.equal(a, a2)),
        "cpu_identical": bool(torch.equal(b_cpu, b2)),
    }
    print("CUDA RNG", out["cuda_rng_roundtrip"], flush=True)
    return out


# ---------------------------------------------------------------------------
# 7. 吞吐参考
# ---------------------------------------------------------------------------


def _bench(fn, iters=5):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def section_throughput() -> dict:
    """中等尺寸（dim 768、12 层、T 1024、batch 8）的训练步（前向 + 反向，BF16 autocast）吞吐。"""
    out: dict = {}
    B, T = 8, 1024
    mc = ModelConfig(
        vocab_size=32768,
        dim=768,
        n_layers=12,
        n_heads=12,
        n_kv_heads=4,
        head_dim=64,
        ffn_dim=2048,
        max_seq_len=T,
    )
    x = torch.randint(0, mc.vocab_size, (B, T + 1), device=CUDA)

    def train_tok_s(m):
        m = m.to(CUDA).train()

        def step():
            m.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = m.loss(x[:, :-1], x[:, 1:])
            loss.backward()

        return round(B * T / _bench(step))

    builds = {
        "dense_gqa": lambda: Transformer(mc),
        "moe_8x_top2(active≈dense)": lambda: moe_transformer(
            mc, MoEConfig.from_model_config(mc, n_experts=8, top_k=2, expert_dim=1024)
        ),
        "mla_kv_lora_256": lambda: mla_mod.mla_transformer(
            ModelConfig(**{**mc.__dict__, "n_kv_heads": 12}),
            mla_mod.MLAConfig.from_model_config(
                ModelConfig(**{**mc.__dict__, "n_kv_heads": 12}), kv_lora_rank=256
            ),
        ),
        "sliding_window_512(1:1)": lambda: sw.convert_to_sliding_window(
            Transformer(mc), sw.make_layer_types(12, global_every=2), window=512
        ),
        "gated_deltanet_hybrid(3:1)": lambda: la.HybridTransformer(
            la.HybridConfig(
                **{k: v for k, v in mc.__dict__.items()},
                full_attention_interval=4,
                linear_kind="gated_deltanet",
                linear_num_key_heads=6,
                linear_num_value_heads=12,
                linear_key_head_dim=64,
                linear_value_head_dim=64,
                linear_chunk_size=64,
            )
        ),
    }
    for name, b in builds.items():
        try:
            torch.manual_seed(0)
            out[name] = {"train_tok_per_s": train_tok_s(b())}
        except Exception as e:  # noqa: BLE001
            out[name] = {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"}
        torch.cuda.empty_cache()
        print("吞吐", name, out[name], flush=True)
    return out


def main() -> None:
    assert torch.cuda.device_count() == 1, "只允许看到 1 张卡（CUDA_VISIBLE_DEVICES=0）"
    torch.set_num_threads(8)
    RESULTS["parity"] = section_parity()
    RESULTS["generation"] = section_generation()
    RESULTS["training_bits"] = section_training_bits()
    RESULTS["throughput"] = section_throughput()
    out = REPO / "out/gpu0-check/parity_cuda.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(RESULTS, ensure_ascii=False, indent=1))
    print(f"写入 {out}")


if __name__ == "__main__":
    main()
