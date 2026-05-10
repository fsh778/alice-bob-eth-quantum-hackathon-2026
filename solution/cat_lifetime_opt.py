#!/usr/bin/env python3
"""
Cat qubit lifetime optimization with CMA-ES (GPU-accelerated).

Fast proxy measurements (no scipy fitting):
  T_z — 5-pt linear fit in the early-time window (TZ_SHORT µs):
        ⟨σ_z(t)⟩ ≈ 1 − t/T_z  →  T_z = −1/slope   (JAX-native LSQ)
  T_x — two-point parity ratio:
        T_x = (t2−t1) / log(P(t1)/P(t2))

All BATCH_SIZE candidates are evaluated in one jit+vmap call,
dispatched as a single GPU kernel launch.

GPU install (CUDA 12):
    pip install "jax[cuda12]" dynamiqs cmaes

Usage:
    python cat_lifetime_opt.py [--epochs N] [--batch B] [--nu NU] [--lam LAM]
    python cat_lifetime_opt.py --epochs 60 --batch 16 --nu 1000 --lam 0.5
"""

import argparse

import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np
from cmaes import SepCMA
from matplotlib import pyplot as plt

# float64 is critical: (1 − ⟨σ_z⟩) can be as small as 1e-4 for large cats
jax.config.update("jax_enable_x64", True)

# ── Fixed system parameters ───────────────────────────────────────────────────
NA = 15  # storage Hilbert-space dimension
NB = 5  # buffer Hilbert-space dimension
KAPPA_B = 10.0  # buffer decay rate       [MHz]
KAPPA_A = 1.0  # storage single-ph. loss [MHz]

# Fast-proxy measurement windows
TZ_SHORT = 10.0  # µs — stays in the linear regime for T_z ≥ 100 µs
TZ_PTS = 5  # points in [0, TZ_SHORT] for linear regression
TX_T1 = 0.30  # µs — first parity sample for T_x
TX_T2 = 1.00  # µs — second parity sample for T_x

_OPTS = dq.Options(progress_meter=False)


# ── JAX-native linear fit helpers ─────────────────────────────────────────────


def _linfit_slope(t, y):
    """Slope of y vs t via closed-form OLS. Both are 1-D JAX arrays."""
    t_m = jnp.mean(t)
    y_m = jnp.mean(y)
    return jnp.sum((t - t_m) * (y - y_m)) / jnp.sum((t - t_m) ** 2)


# ── Loss function factory ─────────────────────────────────────────────────────


def make_loss_fn(target_nu: float, lam: float):
    """
    Return a pure JAX function  loss_fn(x) → (loss, T_x, T_z)
    that is safe to jit and vmap.

    All operators are built once and closed over; they are constant
    JAX arrays (not traced), so vmap sees them as shared constants.
    """
    a = dq.tensor(dq.destroy(NA), dq.eye(NB))
    b = dq.tensor(dq.eye(NA), dq.destroy(NB))

    n_op = a.dag() @ a  # photon-number (storage)
    sx_op = (1j * jnp.pi * n_op).expm()  # parity  (−1)^n

    loss_b = jnp.sqrt(KAPPA_B) * b
    loss_a = jnp.sqrt(KAPPA_A) * a

    # Fixed time grids — constant arrays, safe under vmap
    ts_z = jnp.linspace(0.0, TZ_SHORT, TZ_PTS)
    ts_x = jnp.array([TX_T1, TX_T2])

    def loss_fn(x):
        """x = [Re(g2), Im(g2), Re(eps_d), Im(eps_d)]"""
        g2 = x[0] + 1j * x[1]
        eps_d = x[2] + 1j * x[3]

        # ── Cat-size estimate from adiabatic elimination ──────────────────────
        eps_2 = 2.0 * g2 * eps_d / KAPPA_B
        kappa_2 = 4.0 * jnp.abs(g2) ** 2 / KAPPA_B
        alpha_sq = 2.0 * (eps_2 - KAPPA_A / 4.0) / kappa_2
        # Phase-preserving square root so blobs stay on the cat manifold
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

        #sz_op = dq.tensor(
        #    g_state @ g_state.dag() - e_state @ e_state.dag(),
        #    dq.eye(NB),
        #)

        # Heterodyne replacement for alpha-free T_Z proxy:
        x_op = (a.dag() + a) / 2   # I quadrature, built once alongside a and b

        # ── T_z: 5-point early-time linear fit ───────────────────────────────
        # ⟨σ_z(t)⟩ ≈ 1 − t/T_z  →  slope of (⟨σ_z⟩ − 1) vs t = −1/T_z
        psi_z = dq.tensor(g_state, dq.fock(NB, 0))
        res_z = dq.mesolve(
            #H, [loss_b, loss_a], psi_z, ts_z, exp_ops=[x_op], options=_OPTS
            H, [loss_b, loss_a], psi_z, ts_z, exp_ops=[x_op], options=_OPTS
        )

        #szt = res_z.expects[0].real  # shape (TZ_PTS,)
        xt = res_z.expects[0].real
        # Normalize by the first point to cancel α
        xt_norm = xt / jnp.clip(xt[0], 1e-9, None)   # ≈ exp(−t/T_Z), starts at 1

        #slope = _linfit_slope(ts_z, szt - 1.0)
        slope = _linfit_slope(ts_z, xt_norm - 1.0)   # slope ≈ −1/T_Z
        Tz = jnp.clip(-1.0 / jnp.clip(slope, -1.0, -1e-9), 1.0, 1e9)

        # ── T_x: two-point parity ratio ───────────────────────────────────────
        # P(t) = exp(−t/T_x)  →  T_x = (t2−t1) / log(P(t1)/P(t2))
        psi_x = dq.tensor((g_state + e_state) / jnp.sqrt(2), dq.fock(NB, 0))
        res_x = dq.mesolve(
            H, [loss_b, loss_a], psi_x, ts_x, exp_ops=[sx_op], options=_OPTS
        )
        px1 = jnp.clip(res_x.expects[0, 0].real, 1e-9, 1.0)
        px2 = jnp.clip(res_x.expects[0, 1].real, 1e-9, 1.0)
        Tx = jnp.clip(
            (TX_T2 - TX_T1) / jnp.log(jnp.clip(px1 / px2, 1.0001, 1e9)),
            1e-3,
            1e9,
        )

        # ── Loss: maximize T_z (primary) and T_x (secondary) ─────────────────
        # Using log-lifetimes keeps the landscape well-scaled across decades.
        # High target_nu (≥1000) forces the optimizer into the large-|α|² regime
        # where T_z grows exponentially while T_x decreases only algebraically.
        ratio = Tz / Tx
        loss = -jnp.log(Tz) - jnp.log(Tx) + lam * jnp.abs(target_nu - ratio)
        return loss, Tx, Tz

    return loss_fn


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--epochs", type=int, default=60, help="CMA-ES epochs")
    parser.add_argument("--batch", type=int, default=12, help="Population size")
    parser.add_argument(
        "--nu",
        type=float,
        default=1000.0,
        help="Target bias T_z/T_x (large → push into high-|α|² regime)",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.5,
        help="Bias penalty weight (lower = maximise lifetimes harder)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip plots")
    args = parser.parse_args()

    # ── Device selection ──────────────────────────────────────────────────────
    gpu = [d for d in jax.devices() if d.platform == "gpu"]
    if len(gpu) > 1:
        jax.config.update("jax_default_device", gpu[1])
        print(f"Running on GPU: {gpu[1]}  (gpu[0] reserved)")
    elif len(gpu) == 1:
        jax.config.update("jax_default_device", gpu[0])
        print(f"Running on GPU: {gpu[0]}")
    else:
        print(
            "No GPU detected — running on CPU (set JAX_PLATFORM_NAME=gpu if available)"
        )

    # ── Build batched loss ────────────────────────────────────────────────────
    loss_fn = make_loss_fn(args.nu, args.lam)
    # vmap over first axis (population), jit the whole batch in one call
    batched_loss = jax.jit(jax.vmap(loss_fn))

    # Warm-up compile pass (avoids counting JIT time in epoch 0)
    print("Compiling (one-time JIT warm-up) …")
    _warm = batched_loss(jnp.ones((args.batch, 4)))
    _warm[0].block_until_ready()
    print("Done.\n")

    # ── CMA-ES ───────────────────────────────────────────────────────────────
    # Start near notebook defaults; widen bounds to let the optimizer discover
    # the high-|α|² region (g2 ≈ 1.6, eps_d ≈ 8–12 for large T_z).
    optimizer = SepCMA(
        mean=np.array([1.0, 0.0, 4.0, 0.0]),
        sigma=0.4,
        bounds=np.array(
            [
                [0.5, 20.0],  # Re(g2)   — minimum ~1.58 ensures κ_2 ≥ κ_a
                [-5.5, 5.5],  # Im(g2)
                [0.5, 25.0],  # Re(eps_d) — larger → bigger cat
                [-5.0, 5.0],  # Im(eps_d)
            ]
        ),
        population_size=args.batch,
        seed=42,
    )

    mean_history = []
    loss_history = []
    Tx_history = []
    Tz_history = []
    ratio_history = []

    print(
        f"CMA-ES  |  epochs={args.epochs}  batch={args.batch}"
        f"  ν_target={args.nu}  λ={args.lam}"
    )
    print(
        f"{'Epoch':>5}  {'loss':>9}  {'T_x (µs)':>10}  {'T_z (µs)':>10}  {'bias':>10}"
    )
    print("─" * 55)

    for epoch in range(args.epochs):
        # Sample population and evaluate entire batch in one GPU call
        xs = jnp.array([optimizer.ask() for _ in range(optimizer.population_size)])
        losses, Txs, Tzs = batched_loss(xs)

        # Convert to numpy for CMA-ES (host-side)
        losses_np = np.array(losses)
        Txs_np = np.array(Txs)
        Tzs_np = np.array(Tzs)
        xs_np = np.array(xs)

        optimizer.tell([(xs_np[i], losses_np[i]) for i in range(len(xs_np))])

        mean_history.append(optimizer.mean.copy())
        loss_history.append(float(np.mean(losses_np)))
        Tx_history.append(float(np.mean(Txs_np)))
        Tz_history.append(float(np.mean(Tzs_np)))
        ratio_history.append(float(np.mean(Tzs_np) / max(np.mean(Txs_np), 1e-9)))

        if epoch % 5 == 0:
            print(
                f"{epoch:5d}  {loss_history[-1]:9.3f}  "
                f"{Tx_history[-1]:10.4f}  {Tz_history[-1]:10.2f}  "
                f"{ratio_history[-1]:10.1f}"
            )

    # ── Final evaluation at the CMA-ES mean ──────────────────────────────────
    m = optimizer.mean
    _, Tx_f, Tz_f = loss_fn(jnp.array(m))
    print("\nOptimized parameters:")
    print(f"  g2    = {m[0]:.4f} + {m[1]:.4f}j")
    print(f"  eps_d = {m[2]:.4f} + {m[3]:.4f}j")
    print(f"  T_x   = {float(Tx_f):.4f} µs")
    print(f"  T_z   = {float(Tz_f):.2f} µs")
    print(f"  bias  = {float(Tz_f) / float(Tx_f):.1f}")

    if args.no_plot:
        return

    # ── Plots ─────────────────────────────────────────────────────────────────
    mean_history = np.array(mean_history)
    epochs_arr = np.arange(args.epochs)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(epochs_arr, loss_history)
    axes[0, 0].set(xlabel="Epoch", ylabel="Loss", title="Loss vs Epoch")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(epochs_arr, Tx_history, label="$T_X$")
    axes[0, 1].semilogy(epochs_arr, Tz_history, label="$T_Z$")
    axes[0, 1].set(xlabel="Epoch", ylabel="Lifetime (µs)", title="Lifetimes vs Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].semilogy(epochs_arr, ratio_history, label="actual $T_Z / T_X$")
    axes[1, 0].axhline(
        args.nu, linestyle="--", color="r", label=f"target ν = {args.nu}"
    )
    axes[1, 0].set(xlabel="Epoch", ylabel="Bias  $T_Z / T_X$", title="Bias vs Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs_arr, mean_history[:, 0], label=r"Re($g_2$)")
    axes[1, 1].plot(epochs_arr, mean_history[:, 1], label=r"Im($g_2$)")
    axes[1, 1].plot(epochs_arr, mean_history[:, 2], label=r"Re($\varepsilon_d$)")
    axes[1, 1].plot(epochs_arr, mean_history[:, 3], label=r"Im($\varepsilon_d$)")
    axes[1, 1].set(xlabel="Epoch", ylabel="Parameter", title="Parameter Convergence")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(
        f"CMA-ES Cat Qubit  (ν_target={args.nu}, λ={args.lam})"
        f"  →  T_z={float(Tz_f):.0f} µs, bias={float(Tz_f) / float(Tx_f):.0f}"
    )
    plt.tight_layout()
    plt.savefig("cat_cmaes_results.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
