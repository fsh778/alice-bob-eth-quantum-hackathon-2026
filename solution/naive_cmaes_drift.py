#!/usr/bin/env python3
"""
naive_cmaes_drift.py
====================
Demonstrates that vanilla CMA-ES fails to track slow hardware drift in the
cat qubit stabilization problem.

Drift model (two phases)
------------------------
Phase 1  epochs   0–99   No drift.  CMA-ES converges and σ shrinks.
Phase 2  epochs 100–399  Sinusoidal drift in the buffer drive amplitude ε_d:

    ε_d_physical(epoch) = ε_d_control + DRIFT_AMP * sin(2π · DRIFT_FREQ · epoch)

The optimizer controls [Re(g₂), Im(g₂), Re(ε_d), Im(ε_d)] but is unaware of
the additive drift.  To maintain target performance it must compensate by
adjusting Re(ε_d_control) inversely to the drift — but once CMA-ES has
converged (σ ≪ DRIFT_AMP), it can no longer make sufficiently large steps.

Failure signature
-----------------
After convergence (epoch ~50–80) the step size σ collapses.  When drift
starts at epoch 100, the optimizer is effectively locked near its converged
mean and cannot track swings of ±DRIFT_AMP in ε_d.  The resulting
degradation shows up as: bias η drifting far from target, T_Z collapsing
when ε_d drops (smaller cat), and T_X growing when ε_d rises (larger cat).

Run
---
    python solution/naive_cmaes_drift.py
Saves plots to solution/naive_cmaes_drift_results.png
"""

import pathlib
import numpy as np
import jax
import jax.numpy as jnp
import dynamiqs as dq
from cmaes import SepCMA
from matplotlib import pyplot as plt

jax.config.update("jax_enable_x64", True)

# ── Fixed system parameters ───────────────────────────────────────────────────
NA      = 15      # storage Hilbert-space truncation
NB      = 5       # buffer Hilbert-space truncation
KAPPA_B = 10.0    # buffer decay rate  [MHz]
KAPPA_A = 1.0     # storage 1-ph loss  [MHz]

# ── Drift schedule ────────────────────────────────────────────────────────────
CONVERGENCE_EPOCHS = 100    # drift-free phase so CMA-ES can fully converge
DRIFT_AMP          = 2.0    # ± amplitude of ε_d drift  [same units as ε_d]
DRIFT_FREQ         = 0.01   # cycles per epoch (period = 100 epochs)

# ── Optimization ──────────────────────────────────────────────────────────────
N_EPOCHS   = 400
BATCH_SIZE = 12
ETA_TARGET = 320.0   # target bias  T_Z / T_X
LAM        = 0.5     # weight for bias-penalty term

# ── Proxy measurement windows (JAX-native, no scipy fitting) ──────────────────
TZ_SHORT = 10.0   # µs — linear regime for T_Z ≥ 100 µs
TZ_PTS   = 5      # number of points for the linear slope fit
TX_T1    = 0.30   # µs — first parity sample for T_X
TX_T2    = 1.00   # µs — second parity sample for T_X

_OPTS = dq.Options(progress_meter=False)


# ── JAX-native linear slope (OLS closed form) ─────────────────────────────────
def _linfit_slope(t: jax.Array, y: jax.Array) -> jax.Array:
    t_m = jnp.mean(t)
    y_m = jnp.mean(y)
    return jnp.sum((t - t_m) * (y - y_m)) / jnp.sum((t - t_m) ** 2)


# ── Loss function factory ─────────────────────────────────────────────────────
def make_loss_fn():
    """
    Build the per-candidate loss function, closing over all constant operators.
    Returns loss_fn(x, eps_d_drift) → (loss, T_x, T_z).

    x            = [Re(g₂), Im(g₂), Re(ε_d), Im(ε_d)]  — the optimizer's knobs
    eps_d_drift  = scalar additive drift on Re(ε_d) (unknown to the optimizer)
    """
    a = dq.tensor(dq.destroy(NA), dq.eye(NB))
    b = dq.tensor(dq.eye(NA), dq.destroy(NB))

    n_op  = a.dag() @ a
    sx_op = (1j * jnp.pi * n_op).expm()   # photon-number parity  (-1)^n

    loss_b = jnp.sqrt(KAPPA_B) * b
    loss_a = jnp.sqrt(KAPPA_A) * a

    ts_z = jnp.linspace(0.0, TZ_SHORT, TZ_PTS)
    ts_x = jnp.array([TX_T1, TX_T2])

    def loss_fn(x: jax.Array, eps_d_drift: jax.Array):
        g2    = x[0] + 1j * x[1]
        # The physical drive amplitude the system sees — includes hidden drift
        eps_d = (x[2] + eps_d_drift) + 1j * x[3]

        # Adiabatic-elimination estimate of cat size (used only to seed the
        # initial coherent state; dynamics run on the full two-mode system)
        eps_2    = 2.0 * g2 * eps_d / KAPPA_B
        kappa_2  = 4.0 * jnp.abs(g2) ** 2 / KAPPA_B
        alpha_sq = 2.0 * (eps_2 - KAPPA_A / 4.0) / kappa_2
        # Phase-preserving sqrt: if alpha_sq is complex, take the principal root
        # along the correct branch so the blob stays on the cat manifold
        alpha_est = jnp.sqrt(jnp.abs(alpha_sq)) * jnp.exp(
            1j * jnp.angle(alpha_sq) / 2.0
        )

        H = (
            jnp.conj(g2) * a @ a @ b.dag()
            + g2 * a.dag() @ a.dag() @ b
            - eps_d * b.dag()
            - jnp.conj(eps_d) * b
        )

        g_state = dq.coherent(NA, alpha_est)
        e_state = dq.coherent(NA, -alpha_est)

        sz_op = dq.tensor(
            g_state @ g_state.dag() - e_state @ e_state.dag(),
            dq.eye(NB),
        )

        # ── T_Z proxy: 5-point early-time linear fit ─────────────────────────
        # ⟨σ_z(t)⟩ ≈ 1 − t/T_Z  →  slope of (⟨σ_z⟩ − 1) vs t = −1/T_Z
        psi_z = dq.tensor(g_state, dq.fock(NB, 0))
        res_z = dq.mesolve(
            H, [loss_b, loss_a], psi_z, ts_z, exp_ops=[sz_op], options=_OPTS
        )
        szt   = res_z.expects[0].real
        slope = _linfit_slope(ts_z, szt - 1.0)
        Tz    = jnp.clip(-1.0 / jnp.clip(slope, -1.0, -1e-9), 1.0, 1e9)

        # ── T_X proxy: two-point parity ratio ────────────────────────────────
        # P(t) ≈ exp(−t/T_X)  →  T_X = (t₂−t₁) / log(P(t₁)/P(t₂))
        psi_x = dq.tensor((g_state + e_state) / jnp.sqrt(2), dq.fock(NB, 0))
        res_x = dq.mesolve(
            H, [loss_b, loss_a], psi_x, ts_x, exp_ops=[sx_op], options=_OPTS
        )
        px1 = jnp.clip(res_x.expects[0, 0].real, 1e-9, 1.0)
        px2 = jnp.clip(res_x.expects[0, 1].real, 1e-9, 1.0)
        Tx  = jnp.clip(
            (TX_T2 - TX_T1) / jnp.log(jnp.clip(px1 / px2, 1.0001, 1e9)),
            1e-3, 1e9,
        )

        ratio = Tz / Tx
        loss  = -jnp.log(Tz) - jnp.log(Tx) + LAM * jnp.abs(ETA_TARGET - ratio)
        return loss, Tx, Tz

    return loss_fn


# ── Drift schedule ────────────────────────────────────────────────────────────
def eps_d_drift_at(epoch: int) -> float:
    """
    Returns the additive ε_d drift at the given epoch.
    Zero during the convergence phase; sinusoidal afterwards.
    """
    if epoch < CONVERGENCE_EPOCHS:
        return 0.0
    return DRIFT_AMP * np.sin(2.0 * np.pi * DRIFT_FREQ * epoch)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    loss_fn = make_loss_fn()

    # Batch evaluation: vmap over candidates; drift is shared within each epoch
    @jax.jit
    def batched_loss(xs: jax.Array, eps_d_drift: jax.Array):
        return jax.vmap(lambda x: loss_fn(x, eps_d_drift))(xs)

    # One-time JIT warm-up to exclude compilation from epoch timing
    print("Compiling (one-time JIT warm-up)…")
    _w = batched_loss(jnp.ones((BATCH_SIZE, 4)), jnp.float64(0.0))
    _w[0].block_until_ready()
    print("Done.\n")

    # ── CMA-ES setup ─────────────────────────────────────────────────────────
    # Starting point: g₂=1, ε_d=4 (baseline from the challenge notebook)
    optimizer = SepCMA(
        mean=np.array([1.0, 0.0, 4.0, 0.0]),
        sigma=0.4,
        bounds=np.array([
            [0.5,  6.0],    # Re(g₂)
            [-1.5, 1.5],    # Im(g₂)
            [0.5, 15.0],    # Re(ε_d)
            [-2.0, 2.0],    # Im(ε_d)
        ]),
        population_size=BATCH_SIZE,
        seed=42,
    )

    # ── History buffers ───────────────────────────────────────────────────────
    loss_history   = []
    Tx_history     = []
    Tz_history     = []
    bias_history   = []
    sigma_history  = []
    drift_history  = []
    mean_history   = []

    print(
        f"Running naive CMA-ES for {N_EPOCHS} epochs "
        f"(drift starts at epoch {CONVERGENCE_EPOCHS})\n"
    )
    print(f"{'Epoch':>5}  {'drift':>7}  {'σ':>7}  {'loss':>8}  "
          f"{'T_X (µs)':>10}  {'T_Z (µs)':>10}  {'bias':>8}")
    print("─" * 68)

    for epoch in range(N_EPOCHS):
        drift = eps_d_drift_at(epoch)
        drift_history.append(drift)

        xs = jnp.array([optimizer.ask() for _ in range(optimizer.population_size)])
        losses, Txs, Tzs = batched_loss(xs, jnp.float64(drift))

        losses_np = np.array(losses)
        Txs_np    = np.array(Txs)
        Tzs_np    = np.array(Tzs)

        optimizer.tell([(np.array(xs[i]), losses_np[i]) for i in range(len(xs))])

        mean_loss = float(np.mean(losses_np))
        mean_Tx   = float(np.mean(Txs_np))
        mean_Tz   = float(np.mean(Tzs_np))
        mean_bias = mean_Tz / max(mean_Tx, 1e-9)

        loss_history.append(mean_loss)
        Tx_history.append(mean_Tx)
        Tz_history.append(mean_Tz)
        bias_history.append(mean_bias)
        sigma_history.append(float(optimizer._sigma))
        mean_history.append(optimizer.mean.copy())

        if epoch % 20 == 0 or epoch == CONVERGENCE_EPOCHS - 1:
            print(
                f"{epoch:5d}  {drift:7.3f}  {optimizer._sigma:7.4f}  "
                f"{mean_loss:8.3f}  {mean_Tx:10.4f}  {mean_Tz:10.2f}  "
                f"{mean_bias:8.1f}"
            )

    # ── Derive the "oracle" optimal Re(ε_d) trajectory ───────────────────────
    # To keep ε_d_physical = 4.0 (the no-drift optimum), the optimizer would
    # need to set Re(ε_d_control) = 4.0 − drift.  This is what CMA-ES should
    # track but cannot once σ has collapsed.
    drift_arr       = np.array(drift_history)
    eps_d_oracle    = 4.0 - drift_arr          # ideal control value
    eps_d_mean      = np.array(mean_history)[:, 2]   # what CMA-ES actually set

    # ── Plots ─────────────────────────────────────────────────────────────────
    epochs   = np.arange(N_EPOCHS)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Naive CMA-ES under ε_d amplitude drift  "
        f"(±{DRIFT_AMP}, period={int(1/DRIFT_FREQ)} ep)  —  "
        f"drift starts at epoch {CONVERGENCE_EPOCHS}",
        fontsize=13,
    )

    # ── 1. Bias η ─────────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.axvline(CONVERGENCE_EPOCHS, color="gray", linestyle=":", linewidth=1,
               label="drift onset")
    ax.semilogy(epochs, bias_history, color="steelblue", label="actual η = T_Z/T_X")
    ax.axhline(ETA_TARGET, linestyle="--", color="red",
               label=f"target η = {ETA_TARGET:.0f}")
    ax.set(xlabel="Epoch", ylabel="Bias η  (log scale)",
           title="Bias  η = T_Z / T_X")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── 2. T_Z and T_X lifetimes ──────────────────────────────────────────────
    ax = axes[0, 1]
    ax.axvline(CONVERGENCE_EPOCHS, color="gray", linestyle=":", linewidth=1)
    ax.semilogy(epochs, Tz_history, color="steelblue", label="$T_Z$ (µs)")
    ax.semilogy(epochs, Tx_history, color="orange",    label="$T_X$ (µs)")
    ax.set(xlabel="Epoch", ylabel="Lifetime (µs)  (log scale)",
           title="Lifetimes $T_Z$ and $T_X$")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── 3. Loss ───────────────────────────────────────────────────────────────
    ax = axes[0, 2]
    ax.axvline(CONVERGENCE_EPOCHS, color="gray", linestyle=":", linewidth=1)
    ax.plot(epochs, loss_history, color="firebrick", label="mean loss")
    ax.set(xlabel="Epoch", ylabel="Loss", title="CMA-ES Loss")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── 4. CMA-ES step size σ (convergence indicator) ─────────────────────────
    ax = axes[1, 0]
    ax.axvline(CONVERGENCE_EPOCHS, color="gray", linestyle=":", linewidth=1,
               label="drift onset")
    ax.semilogy(epochs, sigma_history, color="purple")
    ax.set(xlabel="Epoch", ylabel="σ  (log scale)",
           title="CMA-ES Step Size σ\n"
                 "(σ collapses before drift starts → can't track)")
    ax.grid(True, alpha=0.3)

    # ── 5. Re(ε_d) control vs oracle ─────────────────────────────────────────
    ax = axes[1, 1]
    ax.axvline(CONVERGENCE_EPOCHS, color="gray", linestyle=":", linewidth=1)
    ax.plot(epochs, eps_d_oracle, linestyle="--", color="red",
            label="oracle Re(ε_d) = 4 − drift")
    ax.plot(epochs, eps_d_mean, color="steelblue",
            label="CMA-ES mean Re(ε_d)")
    ax.set(xlabel="Epoch", ylabel="Re(ε_d)",
           title="Control vs Oracle\n"
                 "(CMA-ES stuck near 4.0; oracle swings ±{:.1f})".format(DRIFT_AMP))
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── 6. Drift signal ───────────────────────────────────────────────────────
    ax = axes[1, 2]
    ax.axvline(CONVERGENCE_EPOCHS, color="gray", linestyle=":", linewidth=1)
    ax.plot(epochs, drift_arr, color="darkorange", label="ε_d drift (hidden)")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.fill_between(epochs, drift_arr, alpha=0.15, color="darkorange")
    ax.set(xlabel="Epoch", ylabel="Δε_d",
           title=f"Drift Signal  (±{DRIFT_AMP}, hidden from optimizer)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    out_path = pathlib.Path(__file__).parent / "naive_cmaes_drift_results.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plots → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
