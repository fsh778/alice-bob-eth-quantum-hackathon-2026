#!/usr/bin/env python3
"""
Cat qubit parameter sweep + CMA-ES fine-tuning.

Physics-motivated bounds (all derived from system constraints):
  g2   ∈ [1.2, 3.5]  — stabilisation threshold κ_2 ≥ κ_a requires |g2| ≥ 1.58;
                        upper limit keeps κ_2 << κ_b (adiabatic elimination valid)
  eps_d ∈ [2.0, 9.5] — lower limit gives α² > 1; upper limit eps_d < κ_b = 10

Phase 1 — grid sweep over (g2_re, eps_d_re) with im parts = 0.
           Evaluates the full NG×NE grid in ONE batched jit+vmap call.
Phase 2 — CMA-ES fine-tuning starting from the sweep's best point,
           now allowing small imaginary parts for g2 and eps_d.

Results saved to cat_sweep_results.npz  (load with np.load).

Usage:
    python cat_sweep.py                         # defaults: 15×15 grid
    python cat_sweep.py --ng 20 --ne 20         # finer grid
    python cat_sweep.py --epochs 80 --batch 16  # more CMA-ES budget
"""

import argparse
import sys
from pathlib import Path

import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np
from cmaes import SepCMA
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

# Import shared physics / loss infrastructure from the companion script
sys.path.insert(0, str(Path(__file__).parent))
from cat_lifetime_opt import make_loss_fn, _OPTS, KAPPA_A, KAPPA_B, NA, NB

jax.config.update("jax_enable_x64", True)

# ── Physics-motivated parameter bounds ───────────────────────────────────────
#
#   Stabilisation threshold: κ_2 = 4g2²/κ_b ≥ κ_a  →  g2 ≥ √(κ_a κ_b/4) ≈ 1.58
#   Adiabatic elimination:   κ_2 << κ_b  →  g2 << √(κ_b²/4) = κ_b/2 = 5
#   Buffer saturation:       eps_d << κ_b = 10
#   Hilbert-space safety:    α² = eps_d/g2 - correction < NA/2 ≈ 7
#
G2_MIN,    G2_MAX    = 1.2,  3.5    # Re(g2) sweep range
EPSD_MIN,  EPSD_MAX  = 2.0,  9.5   # Re(eps_d) sweep range

# CMA-ES bounds (adds small imaginary freedom for fine-tuning)
BOUNDS = np.array([
    [G2_MIN,   G2_MAX],    # Re(g2)
    [-0.5,      0.5],      # Im(g2)   — near-zero; cat manifold is on real axis
    [EPSD_MIN, EPSD_MAX],  # Re(eps_d)
    [-0.5,      0.5],      # Im(eps_d)
])


def alpha_sq(g2_re: float, eps_d_re: float) -> float:
    """Analytic cat size squared from adiabatic elimination (real knobs)."""
    eps_2   = 2.0 * g2_re * eps_d_re / KAPPA_B
    kappa_2 = 4.0 * g2_re ** 2 / KAPPA_B
    return float(2.0 * (eps_2 - KAPPA_A / 4.0) / kappa_2)


def analytic_lifetimes(g2_re: float, eps_d_re: float):
    """
    Fast analytic estimates (no simulation) — useful for sanity checks.
    T_z ~ exp(2α²)/κ_a,   T_x ~ κ_b/(4g2² α²) = 1/(κ_2 α²)
    """
    a2      = max(alpha_sq(g2_re, eps_d_re), 1e-6)
    kappa_2 = 4.0 * g2_re ** 2 / KAPPA_B
    Tz = np.exp(2.0 * a2) / KAPPA_A
    Tx = 1.0 / (kappa_2 * a2)
    return Tx, Tz


def make_alpha_comparison_fn():
    """
    Return a vmappable JAX function  x → (alpha_sq_theory, alpha_sq_numerical)

    alpha_sq_theory:
        Closed-form result from adiabatic elimination of the buffer mode.
        Assumes κ_b → ∞ (buffer decays instantly) and single-photon loss
        enters only as a small correction:
            α²_AE = 2(ε₂ − κ_a/4) / κ₂,   ε₂ = 2g₂ε_d/κ_b,  κ₂ = 4|g₂|²/κ_b

    alpha_sq_numerical:
        |⟨a⟩|² extracted from the full two-mode Hamiltonian after relaxing to
        steady state.  No approximations — finite κ_b corrections and the true
        buffer back-action on the storage are fully included.

    The ratio  alpha_sq_numerical / alpha_sq_theory  reveals where the
    adiabatic approximation over- or under-estimates the true cat size:
      - ratio < 1: finite κ_b shrinks the cat (buffer not fully eliminated)
      - ratio > 1: single-photon loss correction over-compensated by AE formula

    Steady-state time: at g2_min = 1.2, κ₂ = 0.576 MHz → τ_ss ≈ 1.7 µs.
    We integrate to 12 µs (≈7τ_ss at g2_min, shorter for larger g2).
    """
    a = dq.tensor(dq.destroy(NA), dq.eye(NB))
    b = dq.tensor(dq.eye(NA), dq.destroy(NB))

    loss_b = jnp.sqrt(KAPPA_B) * b
    loss_a = jnp.sqrt(KAPPA_A) * a

    ts_ss = jnp.linspace(0.0, 12.0, 40)   # 40 pts over 12 µs

    def alpha_comparison_fn(x):
        g2    = x[0] + 1j * x[1]
        eps_d = x[2] + 1j * x[3]

        # ── Theory: adiabatic elimination ────────────────────────────────────
        eps_2   = 2.0 * g2 * eps_d / KAPPA_B
        kappa_2 = 4.0 * jnp.abs(g2) ** 2 / KAPPA_B
        alpha_sq_ae = 2.0 * (eps_2 - KAPPA_A / 4.0) / kappa_2
        # Magnitude only (imaginary part is tiny for near-real knobs)
        alpha_sq_theory = jnp.abs(alpha_sq_ae)

        # Phase of AE α (needed to initialise near the correct blob)
        alpha_phase = jnp.angle(alpha_sq_ae) / 2.0
        alpha_init  = jnp.sqrt(alpha_sq_theory) * jnp.exp(1j * alpha_phase)

        # ── Numerical: full Hamiltonian steady state ──────────────────────────
        # Initialise near the AE prediction, then let the full two-mode
        # Lindbladian drive the system to its true steady state.
        # ⟨a⟩_ss = α_actual  (storage is approximately coherent at steady state)
        H = (
            jnp.conj(g2) * a @ a @ b.dag()
            + g2 * a.dag() @ a.dag() @ b
            - eps_d * b.dag()
            - jnp.conj(eps_d) * b
        )
        g_state = dq.coherent(NA, alpha_init)
        psi0    = dq.tensor(g_state, dq.fock(NB, 0))

        res = dq.mesolve(H, [loss_b, loss_a], psi0, ts_ss,
                         exp_ops=[a], options=_OPTS)

        # Last time point is the steady-state ⟨a⟩ of the storage mode
        a_ss               = res.expects[0, -1]   # complex number
        alpha_sq_numerical = jnp.abs(a_ss) ** 2

        return alpha_sq_theory, alpha_sq_numerical

    return alpha_comparison_fn


def run_gradient_optimizer(
    loss_fn,
    x0,
    bounds_arr,
    n_steps: int = 200,
    lr: float = 0.02,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps_adam: float = 1e-8,
    print_every: int = 20,
):
    """
    Adam optimizer with exact gradients via JAX reverse-mode autodiff.

    Advantages over CMA-ES:
      - Gradient is computed analytically through the dynamiqs ODE solver
        (no finite differences, no population of forward passes)
      - JIT-compiled forward+backward in a single kernel launch
      - Each step costs ~2× one forward evaluation; CMA-ES costs BATCH_SIZE
        forward evaluations per epoch

    Bounds are enforced by projecting (clipping) after each Adam update.
    """
    def loss_with_aux(x):
        loss, Tx, Tz = loss_fn(x)
        return loss, (Tx, Tz)

    grad_fn = jax.jit(jax.value_and_grad(loss_with_aux, has_aux=True))

    bounds_lo = jnp.array(bounds_arr[:, 0])
    bounds_hi = jnp.array(bounds_arr[:, 1])

    x = jnp.array(x0, dtype=jnp.float64)
    m = jnp.zeros_like(x)
    v = jnp.zeros_like(x)

    loss_hist, Tx_hist, Tz_hist, param_hist = [], [], [], []

    for t in range(1, n_steps + 1):
        (loss_val, (Tx_val, Tz_val)), grad = grad_fn(x)

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        x = x - lr * m_hat / (jnp.sqrt(v_hat) + eps_adam)
        x = jnp.clip(x, bounds_lo, bounds_hi)

        loss_hist.append(float(loss_val))
        Tx_hist.append(float(Tx_val))
        Tz_hist.append(float(Tz_val))
        param_hist.append(np.array(x))

        if print_every and t % print_every == 0:
            bias = float(Tz_val) / max(float(Tx_val), 1e-9)
            print(
                f"  Step {t:4d}  loss={float(loss_val):.3f}  "
                f"Tx={float(Tx_val):.4f}  Tz={float(Tz_val):.2f}  bias={bias:.1f}"
            )

    return (
        np.array(loss_hist),
        np.array(Tx_hist),
        np.array(Tz_hist),
        np.array(param_hist),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ng",     type=int,   default=15,    help="Grid points for g2")
    parser.add_argument("--ne",     type=int,   default=15,    help="Grid points for eps_d")
    parser.add_argument("--nu",     type=float, default=1000.0,help="Target bias T_z/T_x")
    parser.add_argument("--lam",    type=float, default=0.5,   help="Bias penalty weight")
    parser.add_argument("--epochs", type=int,   default=60,    help="CMA-ES epochs")
    parser.add_argument("--batch",  type=int,   default=12,    help="CMA-ES population size")
    parser.add_argument("--no-plot",    action="store_true",       help="Skip plots")
    parser.add_argument("--grad-steps", type=int,   default=200,   help="Adam gradient steps")
    parser.add_argument("--grad-lr",    type=float, default=0.02,  help="Adam learning rate")
    parser.add_argument("--save",   default="cat_sweep_results.npz",
                        help="Output file for results")
    args = parser.parse_args()

    # ── Device setup ─────────────────────────────────────────────────────────
    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    if len(gpus) > 1:
        jax.config.update("jax_default_device", gpus[1])
        print(f"Running on GPU: {gpus[1]}  (gpu[0] reserved, {len(gpus)} available)")
    elif len(gpus) == 1:
        jax.config.update("jax_default_device", gpus[0])
        print(f"Running on GPU: {gpus[0]}")
    else:
        print("No GPU — running on CPU")

    # ── Build vmapped functions ───────────────────────────────────────────────
    loss_fn          = make_loss_fn(args.nu, args.lam)
    batched_loss     = jax.jit(jax.vmap(loss_fn))
    alpha_fn         = make_alpha_comparison_fn()
    batched_alpha    = jax.jit(jax.vmap(alpha_fn))

    # ── Phase 1: Grid Sweep ───────────────────────────────────────────────────
    g2_vals   = np.linspace(G2_MIN, G2_MAX, args.ng)
    epsd_vals = np.linspace(EPSD_MIN, EPSD_MAX, args.ne)
    G2, EPSD  = np.meshgrid(g2_vals, epsd_vals, indexing="ij")  # (NG, NE)

    # Flatten to (NG*NE, 4) — im parts are 0 for the sweep
    n_pts  = args.ng * args.ne
    xs_flat = np.column_stack([
        G2.ravel(),
        np.zeros(n_pts),
        EPSD.ravel(),
        np.zeros(n_pts),
    ])  # shape (n_pts, 4)

    print(f"\n── Phase 1: {args.ng}×{args.ne} = {n_pts} point grid sweep ──")
    print(f"   g2   ∈ [{G2_MIN}, {G2_MAX}]   (stabilisation threshold ≈ 1.58)")
    print(f"   eps_d ∈ [{EPSD_MIN}, {EPSD_MAX}]   (adiabatic elimination requires < κ_b={KAPPA_B})")
    print("   Compiling …")

    # Warm-up
    _w = batched_loss(jnp.ones((2, 4)))
    _w[0].block_until_ready()

    print("   Evaluating full grid (loss + lifetimes) …")
    losses_flat, Txs_flat, Tzs_flat = batched_loss(jnp.array(xs_flat))
    losses_flat = np.array(losses_flat)
    Txs_flat    = np.array(Txs_flat)
    Tzs_flat    = np.array(Tzs_flat)
    ratios_flat = Tzs_flat / np.clip(Txs_flat, 1e-9, None)

    print("   Evaluating α comparison (theory vs Hamiltonian steady state) …")
    a2_theory_flat, a2_num_flat = batched_alpha(jnp.array(xs_flat))
    a2_theory_flat = np.array(a2_theory_flat)
    a2_num_flat    = np.array(a2_num_flat)
    a2_ratio_flat  = a2_num_flat / np.clip(a2_theory_flat, 1e-9, None)

    # Reshape to (NG, NE) grids
    LOSS     = losses_flat.reshape(args.ng, args.ne)
    TX       = Txs_flat.reshape(args.ng, args.ne)
    TZ       = Tzs_flat.reshape(args.ng, args.ne)
    RATIO    = ratios_flat.reshape(args.ng, args.ne)
    A2       = a2_theory_flat.reshape(args.ng, args.ne)   # theory α²
    A2_NUM   = a2_num_flat.reshape(args.ng, args.ne)      # numerical α²
    A2_RATIO = a2_ratio_flat.reshape(args.ng, args.ne)    # numerical / theory

    # ── Sweep summary ─────────────────────────────────────────────────────────
    best_idx   = np.argmin(losses_flat)
    best_g2    = xs_flat[best_idx, 0]
    best_epsd  = xs_flat[best_idx, 2]
    best_Tx    = Txs_flat[best_idx]
    best_Tz    = Tzs_flat[best_idx]
    best_ratio = ratios_flat[best_idx]
    best_a2    = alpha_sq(best_g2, best_epsd)

    best_a2_num = float(a2_num_flat[best_idx])

    print(f"\n   Best grid point:")
    print(f"     g2    = {best_g2:.3f},  eps_d = {best_epsd:.3f}")
    print(f"     α² (theory / AE)  = {best_a2:.4f}")
    print(f"     α² (numerical)    = {best_a2_num:.4f}   "
          f"(correction = {best_a2_num/max(best_a2,1e-9):.4f}×)")
    print(f"     T_x   = {best_Tx:.4f} µs")
    print(f"     T_z   = {best_Tz:.2f} µs   ({best_Tz/1e3:.2f} ms)")
    print(f"     bias  = {best_ratio:.1f}   (target {args.nu})")

    # Print top-10 sorted by T_z descending
    print(f"\n   Top 10 by T_z (simulated):")
    print(f"   {'g2':>6}  {'eps_d':>7}  {'α²':>6}  {'T_x (µs)':>10}  {'T_z (µs)':>10}  {'bias':>8}  {'loss':>8}")
    print("   " + "─" * 64)
    top_idx = np.argsort(Tzs_flat)[::-1][:10]
    for i in top_idx:
        g2i, ei = xs_flat[i, 0], xs_flat[i, 2]
        print(
            f"   {g2i:6.3f}  {ei:7.3f}  {alpha_sq(g2i,ei):6.3f}"
            f"  {Txs_flat[i]:10.4f}  {Tzs_flat[i]:10.2f}"
            f"  {ratios_flat[i]:8.1f}  {losses_flat[i]:8.3f}"
        )

    # ── Phase 2: CMA-ES fine-tuning ───────────────────────────────────────────
    print(f"\n── Phase 2: CMA-ES fine-tuning from best grid point ──")
    print(f"   Start: g2={best_g2:.3f}, eps_d={best_epsd:.3f}")

    # Tighter sigma: we have a good starting point from the sweep
    mean0  = np.array([best_g2, 0.0, best_epsd, 0.0])
    sigma0 = 0.2

    optimizer = SepCMA(
        mean=mean0,
        sigma=sigma0,
        bounds=BOUNDS,
        population_size=args.batch,
        seed=42,
    )

    opt_mean_history  = []
    opt_loss_history  = []
    opt_Tx_history    = []
    opt_Tz_history    = []
    opt_ratio_history = []

    print(f"{'Epoch':>5}  {'loss':>9}  {'T_x (µs)':>10}  {'T_z (µs)':>11}  {'bias':>10}")
    print("─" * 55)

    for epoch in range(args.epochs):
        xs = jnp.array([optimizer.ask() for _ in range(optimizer.population_size)])
        losses, Txs, Tzs = batched_loss(xs)

        losses_np = np.array(losses)
        Txs_np    = np.array(Txs)
        Tzs_np    = np.array(Tzs)
        xs_np     = np.array(xs)

        optimizer.tell([(xs_np[i], losses_np[i]) for i in range(len(xs_np))])

        opt_mean_history.append(optimizer.mean.copy())
        opt_loss_history.append(float(np.mean(losses_np)))
        opt_Tx_history.append(float(np.mean(Txs_np)))
        opt_Tz_history.append(float(np.mean(Tzs_np)))
        opt_ratio_history.append(float(np.mean(Tzs_np) / max(np.mean(Txs_np), 1e-9)))

        if epoch % 10 == 0:
            print(
                f"{epoch:5d}  {opt_loss_history[-1]:9.3f}  "
                f"{opt_Tx_history[-1]:10.4f}  {opt_Tz_history[-1]:11.2f}  "
                f"{opt_ratio_history[-1]:10.1f}"
            )

    # Final evaluation at CMA-ES mean
    m = optimizer.mean
    _, Tx_opt, Tz_opt = loss_fn(jnp.array(m))
    Tx_opt = float(Tx_opt)
    Tz_opt = float(Tz_opt)

    print(f"\n── Final optimized parameters ──")
    print(f"  g2    = {m[0]:.5f} + {m[1]:.5f}j")
    print(f"  eps_d = {m[2]:.5f} + {m[3]:.5f}j")
    print(f"  α²    = {alpha_sq(m[0], m[2]):.4f}")
    print(f"  T_x   = {Tx_opt:.5f} µs")
    print(f"  T_z   = {Tz_opt:.2f} µs   ({Tz_opt/1e3:.3f} ms)")
    print(f"  bias  = {Tz_opt / Tx_opt:.1f}   (target {args.nu})")

    # ── Phase 3: Gradient-based optimization (Adam + JAX autodiff) ───────────
    print(
        f"\n── Phase 3: Adam gradient optimizer "
        f"({args.grad_steps} steps, lr={args.grad_lr}) ──"
    )
    print(f"   Start: same grid best → g2={best_g2:.3f}, eps_d={best_epsd:.3f}")
    print("   Compiling gradient (one-time JIT warm-up) …")

    x_grad_start = np.array([best_g2, 0.0, best_epsd, 0.0])
    def _scalar_loss(x):
        return loss_fn(x)[0]

    _wg_val, _wg_grad = jax.jit(jax.value_and_grad(_scalar_loss))(jnp.array(x_grad_start))
    _wg_val.block_until_ready()
    print("   Done.\n")

    grad_loss_hist, grad_Tx_hist, grad_Tz_hist, grad_param_hist = run_gradient_optimizer(
        loss_fn,
        x_grad_start,
        BOUNDS,
        n_steps=args.grad_steps,
        lr=args.grad_lr,
    )

    grad_ratio_hist = grad_Tz_hist / np.clip(grad_Tx_hist, 1e-9, None)

    x_grad_final = jnp.array(grad_param_hist[-1])
    _, Tx_grad, Tz_grad = loss_fn(x_grad_final)
    Tx_grad, Tz_grad = float(Tx_grad), float(Tz_grad)

    print("\n── Adam optimizer final result ──")
    print(f"  g2    = {grad_param_hist[-1][0]:.5f} + {grad_param_hist[-1][1]:.5f}j")
    print(f"  eps_d = {grad_param_hist[-1][2]:.5f} + {grad_param_hist[-1][3]:.5f}j")
    print(f"  α²    = {alpha_sq(grad_param_hist[-1][0], grad_param_hist[-1][2]):.4f}")
    print(f"  T_x   = {Tx_grad:.5f} µs")
    print(f"  T_z   = {Tz_grad:.2f} µs   ({Tz_grad / 1e3:.3f} ms)")
    print(f"  bias  = {Tz_grad / max(Tx_grad, 1e-9):.1f}   (target {args.nu})")

    # ── Save results ──────────────────────────────────────────────────────────
    np.savez(
        args.save,
        # sweep grid
        g2_vals=g2_vals, epsd_vals=epsd_vals,
        LOSS=LOSS, TX=TX, TZ=TZ, RATIO=RATIO,
        A2=A2, A2_NUM=A2_NUM, A2_RATIO=A2_RATIO,
        # sweep flat
        xs_flat=xs_flat, losses_flat=losses_flat,
        Txs_flat=Txs_flat, Tzs_flat=Tzs_flat, ratios_flat=ratios_flat,
        a2_theory_flat=a2_theory_flat, a2_num_flat=a2_num_flat,
        # CMA-ES
        cmaes_mean_history=np.array(opt_mean_history),
        cmaes_loss_history=np.array(opt_loss_history),
        cmaes_Tx_history=np.array(opt_Tx_history),
        cmaes_Tz_history=np.array(opt_Tz_history),
        cmaes_ratio_history=np.array(opt_ratio_history),
        # Gradient (Adam)
        grad_loss_history=grad_loss_hist,
        grad_Tx_history=grad_Tx_hist,
        grad_Tz_history=grad_Tz_hist,
        grad_ratio_history=grad_ratio_hist,
        grad_param_history=grad_param_hist,
        # final
        final_params=m,
        final_Tx=Tx_opt, final_Tz=Tz_opt,
        final_grad_params=grad_param_hist[-1],
        final_grad_Tx=Tx_grad, final_grad_Tz=Tz_grad,
    )
    print(f"\n  Results saved to {args.save}")

    if args.no_plot:
        return

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"Cat qubit sweep  (ν_target={args.nu}, λ={args.lam})\n"
        f"Optimal: g2={m[0]:.3f}, ε_d={m[2]:.3f}  →  "
        f"T_z={Tz_opt:.0f} µs, bias={Tz_opt/Tx_opt:.0f}",
        fontsize=12,
    )

    kw_heat = dict(origin="lower", aspect="auto",
                   extent=[EPSD_MIN, EPSD_MAX, G2_MIN, G2_MAX])
    kw_cb   = dict(fraction=0.046, pad=0.04)

    # ── Row 1: sweep heatmaps ─────────────────────────────────────────────────
    ax = fig.add_subplot(2, 4, 1)
    im = ax.imshow(A2, **kw_heat)
    ax.set(title="α²  (adiabatic elim.)", xlabel="ε_d", ylabel="g₂")
    plt.colorbar(im, ax=ax, **kw_cb)
    ax.axhline(np.sqrt(KAPPA_A * KAPPA_B / 4), color="r", lw=1.2,
               linestyle="--", label="κ₂=κₐ threshold")
    ax.legend(fontsize=7)

    ax = fig.add_subplot(2, 4, 2)
    im = ax.imshow(TX, norm=LogNorm(), **kw_heat, cmap="plasma_r")
    ax.set(title="T_x  (µs)", xlabel="ε_d", ylabel="g₂")
    plt.colorbar(im, ax=ax, **kw_cb)

    ax = fig.add_subplot(2, 4, 3)
    im = ax.imshow(TZ, norm=LogNorm(), **kw_heat, cmap="viridis")
    ax.set(title="T_z  (µs)", xlabel="ε_d", ylabel="g₂")
    plt.colorbar(im, ax=ax, **kw_cb)
    ax.plot(best_epsd, best_g2, "r*", ms=12, label="sweep best")
    ax.plot(m[2], m[0], "w^", ms=10, label="CMA-ES opt")
    ax.legend(fontsize=7)

    ax = fig.add_subplot(2, 4, 4)
    # Clamp ratio for display so log scale doesn't explode
    RATIO_clamped = np.clip(RATIO, 1, 1e5)
    im = ax.imshow(RATIO_clamped, norm=LogNorm(), **kw_heat, cmap="coolwarm")
    ax.axhline(np.sqrt(KAPPA_A * KAPPA_B / 4), color="k", lw=1.2, linestyle="--")
    ax.set(title="Bias  T_z / T_x", xlabel="ε_d", ylabel="g₂")
    plt.colorbar(im, ax=ax, **kw_cb)

    # ── Row 2: CMA-ES convergence ─────────────────────────────────────────────
    opt_mean_history = np.array(opt_mean_history)
    ep = np.arange(args.epochs)

    ax = fig.add_subplot(2, 4, 5)
    ax.plot(ep, opt_loss_history)
    ax.set(xlabel="Epoch", ylabel="Loss", title="CMA-ES loss")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(2, 4, 6)
    ax.semilogy(ep, opt_Tx_history, label="$T_X$")
    ax.semilogy(ep, opt_Tz_history, label="$T_Z$")
    ax.set(xlabel="Epoch", ylabel="Lifetime (µs)", title="Lifetimes")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(2, 4, 7)
    ax.semilogy(ep, opt_ratio_history, label="bias")
    ax.axhline(args.nu, color="r", linestyle="--", label=f"target {args.nu}")
    ax.set(xlabel="Epoch", ylabel="T_z / T_x", title="Bias convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(2, 4, 8)
    ax.plot(ep, opt_mean_history[:, 0], label=r"Re($g_2$)")
    ax.plot(ep, opt_mean_history[:, 1], label=r"Im($g_2$)")
    ax.plot(ep, opt_mean_history[:, 2], label=r"Re($\varepsilon_d$)")
    ax.plot(ep, opt_mean_history[:, 3], label=r"Im($\varepsilon_d$)")
    ax.set(xlabel="Epoch", ylabel="Value", title="Parameter evolution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig("cat_sweep_results.png", dpi=150)

    # ── Figure 2: α comparison — theory vs numerical vs ratio ─────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
    fig2.suptitle("α² comparison: adiabatic elimination vs Hamiltonian steady state",
                  fontsize=12)

    vmax_a2 = max(A2.max(), A2_NUM.max())

    ax = axes2[0]
    im = ax.imshow(A2, origin="lower", aspect="auto", vmin=0, vmax=vmax_a2,
                   extent=[EPSD_MIN, EPSD_MAX, G2_MIN, G2_MAX], cmap="magma")
    ax.set(title="α²  theory  (adiabatic elim.)", xlabel="ε_d", ylabel="g₂")
    ax.axhline(np.sqrt(KAPPA_A * KAPPA_B / 4), color="w", lw=1.2, linestyle="--",
               label="κ₂=κₐ")
    ax.legend(fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes2[1]
    im = ax.imshow(A2_NUM, origin="lower", aspect="auto", vmin=0, vmax=vmax_a2,
                   extent=[EPSD_MIN, EPSD_MAX, G2_MIN, G2_MAX], cmap="magma")
    ax.set(title="α²  numerical  (Hamiltonian steady state)", xlabel="ε_d", ylabel="g₂")
    ax.axhline(np.sqrt(KAPPA_A * KAPPA_B / 4), color="w", lw=1.2, linestyle="--",
               label="κ₂=κₐ")
    ax.legend(fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes2[2]
    # Ratio centred on 1: blue < 1 (AE over-estimates), red > 1 (under-estimates)
    ratio_min = float(np.nanmin(A2_RATIO))
    ratio_max = float(np.nanmax(A2_RATIO))
    vcenter   = 1.0
    vlo = min(ratio_min, 2 * vcenter - ratio_max)  # symmetric around 1
    vhi = max(ratio_max, 2 * vcenter - ratio_min)
    norm_ratio = TwoSlopeNorm(vmin=vlo, vcenter=vcenter, vmax=vhi)
    im = ax.imshow(A2_RATIO, origin="lower", aspect="auto", norm=norm_ratio,
                   extent=[EPSD_MIN, EPSD_MAX, G2_MIN, G2_MAX], cmap="RdBu_r")
    ax.set(title="α²_num / α²_theory\n(1 = AE exact; <1 = AE over-estimates)",
           xlabel="ε_d", ylabel="g₂")
    ax.axhline(np.sqrt(KAPPA_A * KAPPA_B / 4), color="k", lw=1.2, linestyle="--",
               label="κ₂=κₐ")
    ax.plot(m[2], m[0], "k^", ms=10, label="CMA-ES opt")
    ax.legend(fontsize=8)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("ratio")

    fig2.tight_layout()
    fig2.savefig("cat_alpha_comparison.png", dpi=150)

    # ── Figure 3: CMA-ES vs Adam comparison ───────────────────────────────────
    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8))
    fig3.suptitle(
        "Optimizer comparison: CMA-ES (derivative-free) vs Adam (JAX autodiff)\n"
        f"Same start: g2={best_g2:.3f}, ε_d={best_epsd:.3f}  |  "
        f"CMA-ES batch={args.batch}, Adam lr={args.grad_lr}",
        fontsize=11,
    )

    cmaes_ep = np.arange(args.epochs)
    grad_ep  = np.arange(args.grad_steps)

    ax = axes3[0, 0]
    ax.plot(cmaes_ep, opt_loss_history,  label=f"CMA-ES ({args.epochs} epochs)")
    ax.plot(grad_ep,  grad_loss_hist,    label=f"Adam ({args.grad_steps} steps)", linestyle="--")
    ax.set(xlabel="Step / Epoch", ylabel="Loss", title="Loss convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes3[0, 1]
    ax.semilogy(cmaes_ep, opt_Tz_history, label="CMA-ES $T_z$")
    ax.semilogy(grad_ep,  grad_Tz_hist,   label="Adam $T_z$", linestyle="--")
    ax.annotate(f"CMA-ES: {opt_Tz_history[-1]:.1f} µs", xy=(0.97, 0.10),
                xycoords="axes fraction", ha="right", fontsize=8, color="C0")
    ax.annotate(f"Adam:   {grad_Tz_hist[-1]:.1f} µs", xy=(0.97, 0.03),
                xycoords="axes fraction", ha="right", fontsize=8, color="C1")
    ax.set(xlabel="Step / Epoch", ylabel="T_z (µs)", title="Bit-flip lifetime T_z")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes3[1, 0]
    ax.semilogy(cmaes_ep, opt_Tx_history, label="CMA-ES $T_x$")
    ax.semilogy(grad_ep,  grad_Tx_hist,   label="Adam $T_x$", linestyle="--")
    ax.annotate(f"CMA-ES: {opt_Tx_history[-1]:.4f} µs", xy=(0.97, 0.10),
                xycoords="axes fraction", ha="right", fontsize=8, color="C0")
    ax.annotate(f"Adam:   {grad_Tx_hist[-1]:.4f} µs", xy=(0.97, 0.03),
                xycoords="axes fraction", ha="right", fontsize=8, color="C1")
    ax.set(xlabel="Step / Epoch", ylabel="T_x (µs)", title="Phase-flip lifetime T_x")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes3[1, 1]
    ax.semilogy(cmaes_ep, opt_ratio_history, label="CMA-ES bias")
    ax.semilogy(grad_ep,  grad_ratio_hist,   label="Adam bias", linestyle="--")
    ax.axhline(args.nu, color="r", linestyle=":", label=f"target ν={args.nu:.0f}")
    ax.set(xlabel="Step / Epoch", ylabel="T_z / T_x", title="Bias convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3.savefig("cat_grad_comparison.png", dpi=150)

    plt.show()


if __name__ == "__main__":
    main()
