"""CUDA parity check of the small modules in zero that are marked "not verified on a GPU yet".

Do they run on CUDA (FP32 / BF16 autocast), and how much do their results differ from CPU FP32?
The test uses one RTX 3090.

    CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/parity_cuda.py

Contents:
1. The main-line Transformer and the experimental modules of Part 5 (MoE, MLA, sliding window,
   hybrid linear attention ×2, MTP): with the same weights and the same batch of tokens, one forward +
   backward pass each on CPU FP32 / CUDA FP32 / CUDA BF16 autocast. The script reports the relative
   error ‖a−b‖/‖b‖ of the output and of the gradients (CPU FP32 is the reference);
2. Cached generation: is the greedy output of each module on CUDA "with cache" token-for-token identical
   to "recompute at each step"? Is greedy speculative decoding / MTP self-speculation on CUDA
   token-for-token identical to greedy decoding of the target model?
3. Activation checkpointing on CUDA: are the gradients the same as without it?
4. The parameter difference between fused AdamW (trainer.py turns it on by default on CUDA) and
   foreach AdamW / CPU AdamW;
5. Muon: the difference between NS5 in CUDA BF16 and in CPU FP32, and the parameter difference of
   MuonAdamW on CUDA and on CPU after a few steps;
6. Save and restore of the CUDA random number state in checkpoint.py;
7. The throughput of some modules on CUDA (reference numbers that show only "it runs";
   this is not a performance check).
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
# 1. Parity check of forward + backward
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
        except Exception as e:  # noqa: BLE001 - record the failure; do not stop the full script
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
    # Main-line Transformer (control group: it shows the size of the error that BF16 alone causes)
    rows.append(
        parity("Transformer (main-line, GQA)", lambda: Transformer(base_cfg()), lm_loss, lm_out)
    )

    # MoE: 8 routed experts with top-2 + 1 shared expert; layer 0 is dense
    def build_moe():
        mc = base_cfg()
        cfg = MoEConfig.from_model_config(
            mc, n_experts=8, top_k=2, expert_dim=96, n_shared_experts=1, aux_loss_coef=0.01
        )
        return moe_transformer(mc, cfg, first_dense=1)

    rows.append(parity("MoE (top-2 of 8 + shared expert, loop over experts)", build_moe, lm_loss, lm_out))

    # MLA (no absorption during training)
    def build_mla(q_lora=None):
        mc = base_cfg(n_kv_heads=4)
        cfg = mla_mod.MLAConfig.from_model_config(mc, kv_lora_rank=64, q_lora_rank=q_lora)
        return mla_mod.mla_transformer(mc, cfg)

    rows.append(parity("MLA", build_mla, lm_loss, lm_out))
    rows.append(parity("MLA (q_lora_rank=64)", lambda: build_mla(64), lm_loss, lm_out))

    # Sliding window: local-global 1:1, window 32
    def build_sw():
        m = Transformer(base_cfg())
        sw.convert_to_sliding_window(m, sw.make_layer_types(4, global_every=2), window=32)
        return m

    rows.append(parity("Sliding window (window 32, local:global = 1:1)", build_sw, lm_loss, lm_out))

    # Hybrid linear attention (3:1)
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
            "Gated DeltaNet hybrid (3:1)", lambda: build_hybrid("gated_deltanet"), lm_loss, lm_out
        )
    )
    rows.append(parity("Linear attention hybrid (3:1)", lambda: build_hybrid("linear"), lm_loss, lm_out))

    # MTP (depth 1)
    rows.append(
        parity(
            "MTP (depth 1)",
            lambda: MTPTransformer(base_cfg(), n_mtp=1),
            lambda m, x: mtp_loss(m, x[:, :-1], x[:, 1:])[0],
            lambda m, x: m(x[:, :-1])[0],
        )
    )
    return rows


# ---------------------------------------------------------------------------
# 2. Cached generation / speculative decoding (on CUDA)
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

    check("Main-line KV cache: with cache = recompute at each step", main_kv)

    def mla_gen():
        mc = base_cfg(n_kv_heads=4, init_std=0.2)
        cfg = mla_mod.MLAConfig.from_model_config(mc, kv_lora_rank=64)
        torch.manual_seed(1)
        m = mla_mod.mla_transformer(mc, cfg).to(CUDA)
        a = mla_mod.generate_greedy(m, prompt, 60, cfg, use_cache=True)
        b = mla_mod.generate_greedy(m, prompt, 60, cfg, use_cache=False)
        return a == b, {"n": len(a)}

    check("MLA: absorption + latent-vector cache = recompute at each step", mla_gen)

    def sw_gen():
        torch.manual_seed(2)
        m = Transformer(base_cfg(init_std=0.2)).to(CUDA)
        sw.convert_to_sliding_window(m, sw.make_layer_types(4, global_every=2), window=16)
        cache = sw.SlidingWindowKVCache.from_model(m, batch_size=1, max_seq_len=128)
        a = sw.generate_greedy(m, prompt, 80, cache=cache)
        b = sw.generate_greedy(m, prompt, 80, cache=None)
        return a == b, {"n": len(a), "cache_device": str(cache.k[0].device)}

    check("Sliding window: ring buffer = recompute at each step", sw_gen)

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

        check(f"Hybrid {kind}: recurrent state = recompute at each step", hy_gen)

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

    check("Speculative decoding: greedy = target-model greedy (1-layer draft / draft = target)", spec)

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

    check("MTP self-speculation: greedy = main-model greedy", mtp_gen)
    return out


# ---------------------------------------------------------------------------
# 3–6. Activation checkpointing, fused AdamW, Muon, CUDA random number state
# ---------------------------------------------------------------------------


def section_training_bits() -> dict:
    out: dict = {}
    # 3. Activation checkpointing
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
    print("Activation checkpointing (CUDA) on/off, relative gradient difference", res, flush=True)

    # 4. fused AdamW vs foreach AdamW vs CPU AdamW: the same sequence of gradients, 20 steps
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

    # 5. Muon: NS5 in CUDA BF16 vs CPU FP32
    from zero.train.muon import build_muon_optimizer, zeropower_via_newtonschulz5

    ns = []
    for shape in ((1280, 1280), (2048, 1280), (1280, 3584), (3584, 1280)):
        torch.manual_seed(0)
        G = torch.randn(*shape)
        ref = zeropower_via_newtonschulz5(G)  # CPU: FP32 by default
        g_cuda = G.to(CUDA)
        cu = zeropower_via_newtonschulz5(g_cuda)  # CUDA: BF16 by default
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

    # MuonAdamW (build_muon_optimizer, the same as in trainer) does 5 steps on CUDA and 5 steps on CPU.
    # Compare the total update of the parameters.
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
    print("MuonAdamW 5 steps", out["muon_optimizer_5_steps"], flush=True)

    # 6. CUDA random number state: round trip of rng_state() / set_rng_state()
    torch.cuda.manual_seed_all(123)
    _ = torch.rand(10, device=CUDA)
    st = rng_state()
    a = torch.rand(1000, device=CUDA)
    b_cpu = torch.rand(10)
    torch.rand(12345, device=CUDA)  # Change the state
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
# 7. Throughput reference
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
    """Throughput of a training step (forward + backward, BF16 autocast) at medium size.

    The size is dim 768, 12 layers, T 1024, batch 8.
    """
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
        print("Throughput", name, out[name], flush=True)
    return out


def main() -> None:
    assert torch.cuda.device_count() == 1, "Only 1 GPU may be visible (CUDA_VISIBLE_DEVICES=0)"
    torch.set_num_threads(8)
    RESULTS["parity"] = section_parity()
    RESULTS["generation"] = section_generation()
    RESULTS["training_bits"] = section_training_bits()
    RESULTS["throughput"] = section_throughput()
    out = REPO / "out/gpu0-check/parity_cuda.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(RESULTS, ensure_ascii=False, indent=1))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
