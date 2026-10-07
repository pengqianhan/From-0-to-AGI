"""Chapter 12 · Minimal code 2: what the Chinchilla formula tells us. How to divide compute between model size and data size.

Chinchilla (Hoffmann et al. 2022) fitted this formula to hundreds of training runs:
    L(N, D) = E + A / N^α + B / D^β
E is the uncertainty of the data itself: a larger model cannot decrease it.
A/N^α is the cost of a model that is too small. B/D^β is the cost of too little data.
For a given compute C = 6ND, put D = C/(6N) into the formula and find the N with the minimum loss.
The result is the "compute-optimal" allocation.

This script trains nothing. It only does arithmetic with two published sets of coefficients.
The absolute loss values apply only to their data and tokenizer, not to our model.
What we can use is the "shape": the optimal ratio and the cost of overtraining.
  - Hoffmann et al. 2022, original paper (Approach 3): E=1.69, A=406.4, B=410.7, α=0.34, β=0.28
  - Besiroglu et al. 2024 (Epoch AI replication, arXiv:2404.10102): E=1.8172, A=482.01, B=2085.43, α=0.3478, β=0.3658
    The replication found that the original coefficients have a bias, because the optimizer stopped too early.
    The replicated coefficients agree with the ratio that Chinchilla used: about 20 tokens per parameter.

Run: uv run python chapters/12-scaling-laws/code/02_chinchilla.py   (less than 1 second)
"""

import numpy as np

FITS = {
    "Hoffmann 2022": dict(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28),
    "Epoch 2024 复现": dict(E=1.8172, A=482.01, B=2085.43, alpha=0.3478, beta=0.3658),
}
# Display names. The keys stay as they are because video/scenes.py looks up FITS by key.
FITS_EN = {"Epoch 2024 复现": "Epoch 2024 replication"}


def loss(N, D, E, A, B, alpha, beta):
    return E + A / N**alpha + B / D**beta


def compute_optimal(C, E, A, B, alpha, beta):
    """Closed-form solution: N_opt = G · (C/6)^(β/(α+β)), with G = (αA / βB)^(1/(α+β))."""
    G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
    N = G * (C / 6) ** (beta / (alpha + beta))
    return N, C / (6 * N)


def compute_optimal_grid(C, fit):
    """Numerical check: scan N along the curve of equal compute and take the minimum loss (the bottom of the IsoFLOP parabola)."""
    Ns = np.logspace(6, 13, 20001)
    Ls = loss(Ns, C / (6 * Ns), **fit)
    i = int(np.argmin(Ls))
    return Ns[i], C / (6 * Ns[i])


def compute_to_reach(target_loss, fit):
    """The compute that the compute-optimal frontier needs to reach target_loss (bisection)."""
    lo, hi = 1e15, 1e30
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        N, D = compute_optimal(mid, **fit)
        lo, hi = (mid, hi) if loss(N, D, **fit) > target_loss else (lo, mid)
    return hi


def main():
    fit = FITS["Epoch 2024 复现"]
    print("① Compute-optimal allocation (Epoch replication coefficients; the grid search agrees with the closed-form solution)")
    print(f"{'compute C':>9} | {'N_opt':>9} {'D_opt':>9} {'tok/param':>9} | {'N_grid':>9}")
    for C in [1e19, 1e21, 1.65e21, 1e23, 1e25]:
        N, D = compute_optimal(C, **fit)
        Ng, _ = compute_optimal_grid(C, fit)
        print(f"{C:>9.2e} | {N / 1e9:>8.2f}B {D / 1e9:>8.1f}B {D / N:>9.1f} | {Ng / 1e9:>8.2f}B")
    for name, f in FITS.items():
        N, D = compute_optimal(1e23, **f)
        print(f"  {FITS_EN.get(name, name)}: at C=1e23, the optimum is {D / N:.0f} tokens/parameter")

    # Main-line pretraining: 689.5M parameters × 400B tokens (see 05_plan_budget.py). Chinchilla uses the
    # rough count C = 6ND, so we use it here too. The exact count in 01_flops.py includes the attention
    # term and gives 2.78e21. The difference is the attention on long sequences of 4096 tokens.
    N_ours, D_ours = 689.5e6, 400e9
    C = 6 * N_ours * D_ours
    N_opt, D_opt = compute_optimal(C, **fit)
    L_ours, L_opt = loss(N_ours, D_ours, **fit), loss(N_opt, D_opt, **fit)
    print(f"\n② Our budget (C ≈ {C:.3g}, rough count 6ND):")
    print(f"  Compute-optimal:  N = {N_opt / 1e9:.2f}B, D = {D_opt / 1e9:.0f}B → L = {L_opt:.4f}")
    print(f"  Main-line choice: N = 0.69B, D = {D_ours / 1e9:.0f}B ({D_ours / N_ours:.0f} tokens/parameter) → L = {L_ours:.4f}")
    C_eq = compute_to_reach(L_ours, fit)
    print(f"  Cost of overtraining: the loss is higher by {L_ours - L_opt:.4f} ({(L_ours - L_opt) / L_opt:.1%}); "
          f"for the same loss, the optimal allocation needs only {C_eq / C:.0%} of the compute")

    print("\n③ Why overtrain: add inference to the total compute (the method of Sardana & Frankle 2023)")
    print("  Goal: reach the same loss. Total compute = training 6ND + inference 2N × tokens generated in the model's lifetime")
    L_target = L_ours
    print(f"  Target loss = {L_target:.4f} (the main-line loss above)")
    print(f"{'gen tokens':>10} | {'cheapest N':>12} {'D':>9} {'tok/param':>9} {'total C':>10}")
    Ns = np.logspace(8, 11, 3001)
    for D_inf in [0, 1e12, 1e13, 1e14]:
        best = None
        for N in Ns:
            gap = L_target - fit["E"] - fit["A"] / N ** fit["alpha"]
            if gap <= 0:
                continue  # this N is too small: no amount of data reaches the target
            D = (fit["B"] / gap) ** (1 / fit["beta"])
            total = 6 * N * D + 2 * N * D_inf
            if best is None or total < best[0]:
                best = (total, N, D)
        total, N, D = best
        print(f"{D_inf:>10.0e} | {N / 1e9:>11.2f}B {D / 1e9:>8.0f}B {D / N:>9.0f} {total:>10.3g}")
    print("  → The more tokens a model must serve, the smaller it should be and the longer it should train. "
          "This is why small models are \"overtrained\".")

    print("\n④ Tokens per parameter of real models (sources in the README)")
    for name, n, d in [
        ("Chinchilla 70B", 70e9, 1.4e12),
        ("Llama 3 8B", 8e9, 15e12),
        ("Puro-2B", 2.0e9, 1.4e12),
        ("MobileLLM-R1-950M", 0.949e9, 4.2e12),
        ("Qwen3-0.6B", 0.6e9, 36e12),
        ("Main line (plan)", 0.6895e9, 400e9),
    ]:
        print(f"  {name:<18} {d / n:>8,.0f} tokens/parameter (about {d / n / 20:,.0f} × 20)")


if __name__ == "__main__":
    main()
