#!/usr/bin/env python3
"""
PPO-only cat qubit stabilization demo.

Runs the same four phases as cat_online_stab.py, but with only the PPO
optimizer so the output is focused and fast to inspect.

  Phase A  Static Adam finds a good operating point (x0).
  Phase B  Freeze x0, evaluate under drift → shows degradation baseline.
  Phase C  Online PPO continuously adjusts x0 to compensate drift.
  Plots    Three figures saved to disk:
             cat_ppo_drift.png    — drift model visualisation
             cat_ppo_baseline.png — static vs PPO lifetimes
             cat_ppo_detail.png   — effective params + tracking error

Usage:
    python cat_ppo_only.py
    python cat_ppo_only.py --n-steps 600 --drift-period 300
    python cat_ppo_only.py --no-plot
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from cat_lifetime_opt import make_loss_fn

jax.config.update("jax_enable_x64", True)

# ── System / drift constants ───────────────────────────────────────────────────
DRIFT_PERIOD  = 300
DRIFT_AMP_G2  = 0.25
DRIFT_AMP_EPS = 0.60
DRIFT_PHASE   = np.pi / 3
OU_TAU        = 30
OU_STD_G2     = 0.030
OU_STD_EPS    = 0.080

BOUNDS = np.array([[1.2, 3.5], [-0.5, 0.5], [2.0, 9.5], [-0.5, 0.5]])
BOUNDS_LO = jnp.array(BOUNDS[:, 0])
BOUNDS_HI = jnp.array(BOUNDS[:, 1])


# ── Drift generation ───────────────────────────────────────────────────────────

def generate_drift(n_steps, seed=0, period=DRIFT_PERIOD):
    rng = np.random.default_rng(seed)
    ou_g2 = ou_eps = 0.0
    dt = 1.0 / OU_TAU
    traj = np.zeros((n_steps, 4))
    for i in range(n_steps):
        slow_g2  = DRIFT_AMP_G2  * np.sin(2 * np.pi * i / period)
        slow_eps = DRIFT_AMP_EPS * np.sin(2 * np.pi * i / period + DRIFT_PHASE)
        ou_g2  = ou_g2  * (1 - dt) + OU_STD_G2  * rng.standard_normal()
        ou_eps = ou_eps * (1 - dt) + OU_STD_EPS * rng.standard_normal()
        traj[i, 0] = slow_g2 + ou_g2
        traj[i, 2] = slow_eps + ou_eps
    return traj


def sinusoidal_only(n_steps, period=DRIFT_PERIOD):
    t = np.arange(n_steps)
    traj = np.zeros((n_steps, 4))
    traj[:, 0] = DRIFT_AMP_G2  * np.sin(2 * np.pi * t / period)
    traj[:, 2] = DRIFT_AMP_EPS * np.sin(2 * np.pi * t / period + DRIFT_PHASE)
    return traj


# ── Phase A: static Adam ───────────────────────────────────────────────────────

def run_static_adam(loss_fn, x0, n_steps=120, lr=0.02):
    def loss_with_aux(x):
        loss, Tx, Tz = loss_fn(x)
        return loss, (Tx, Tz)

    grad_fn = jax.jit(jax.value_and_grad(loss_with_aux, has_aux=True))
    x = jnp.array(x0, dtype=jnp.float64)
    m = v = jnp.zeros(4)

    for t in range(1, n_steps + 1):
        (lv, (Tv, Zv)), g = grad_fn(x)
        m = 0.9  * m + 0.1   * g
        v = 0.999 * v + 0.001 * g**2
        x = x - lr * (m / (1 - 0.9**t)) / (jnp.sqrt(v / (1 - 0.999**t)) + 1e-8)
        x = jnp.clip(x, BOUNDS_LO, BOUNDS_HI)
        if t % 30 == 0:
            print(f"    step {t:4d}  loss={float(lv):.3f}  "
                  f"Tz={float(Zv):.1f} µs  Tx={float(Tv):.4f} µs")

    return np.array(x)


# ── Phase B: static evaluation under drift ────────────────────────────────────

def evaluate_static(batched_loss, x_fixed, drift_traj):
    x_all = np.clip(x_fixed[None, :] + drift_traj, BOUNDS[:, 0], BOUNDS[:, 1])
    _, Txs, Tzs = batched_loss(jnp.array(x_all))
    return np.array(Txs), np.array(Tzs)


# ── PPO helpers ────────────────────────────────────────────────────────────────

def _init_mlp(key, in_dim, hidden, out_dim):
    params = []
    for fan_in, fan_out in [(in_dim, hidden), (hidden, hidden), (hidden, out_dim)]:
        key, sub = jax.random.split(key)
        W = jax.random.normal(sub, (fan_in, fan_out)) * jnp.sqrt(2.0 / fan_in)
        b = jnp.zeros(fan_out)
        params.append((W, b))
    return params


def _mlp(params, x):
    for i, (W, b) in enumerate(params):
        x = x @ W + b
        if i < len(params) - 1:
            x = jnp.tanh(x)
    return x


def _adam_pytree(params, grads, m, v, t, lr, b1=0.9, b2=0.999, eps=1e-8):
    m = jax.tree_util.tree_map(lambda mi, gi: b1 * mi + (1 - b1) * gi, m, grads)
    v = jax.tree_util.tree_map(lambda vi, gi: b2 * vi + (1 - b2) * gi**2, v, grads)
    mh = jax.tree_util.tree_map(lambda mi: mi / (1 - b1**t), m)
    vh = jax.tree_util.tree_map(lambda vi: vi / (1 - b2**t), v)
    params = jax.tree_util.tree_map(
        lambda p, mi, vi: p - lr * mi / (jnp.sqrt(vi) + eps), params, mh, vh)
    return params, m, v


def _gae(rewards, values, next_value, gamma, lam):
    T, adv, gae = len(rewards), np.zeros(len(rewards)), 0.0
    for t in reversed(range(T)):
        nv    = next_value if t == T - 1 else values[t + 1]
        gae   = rewards[t] + gamma * nv - values[t] + gamma * lam * gae
        adv[t] = gae
    return adv, adv + values


# ── Phase C: online PPO ────────────────────────────────────────────────────────

def run_online_ppo(loss_fn, x0, drift_traj,
                   lr_actor=3e-4, lr_critic=1e-3,
                   optimizer_freq=3, horizon=64, n_ppo_epochs=4,
                   clip_eps=0.2, gamma=0.99, gae_lambda=0.95,
                   entropy_coef=0.01, action_scale=0.25,
                   hidden_size=64, seed=7,
                   nu=1000.0, lam=0.5):
    """
    Online PPO with JAX-JIT compiled actor+critic gradient.

    Bugs fixed vs original:
      1. Log-std tightened from (-4,-0.5) to (-3,-2): max std 0.61→0.14.
         The original allowed noise ≫ action_scale, swamping the tanh-bounded
         mean and creating massive distribution mismatch in the PPO ratio.
      2. Bias penalty capped at 20 in the reward: the uncapped |Tz/Tx - ν|
         term (potentially thousands) completely drowned log(Tz) during
         exploration, causing the policy to optimize ratio satisfaction
         at the expense of Tz — manifesting as "maximize Tx."
      3. State trend signals replace the slow reward EMA: delta_log_Tz and
         delta_log_Tx give the policy direct per-step feedback on whether
         Tz/Tx are improving, enabling drift-direction inference without
         observing drift directly.

    State (8-D): normalized x_nominal (4), log Tz (1), log Tx (1),
                 Δlog Tz (1), Δlog Tx (1).
    Action (4-D): Δx sampled from N(action_scale·tanh(μ), exp(log_std)²)
                  with log_std ∈ [-3, -2] → std ∈ [0.05, 0.14].
    Reward: log(Tz) + log(Tx) - lam·clip(|Tz/Tx - ν|, 0, 20)
            — penalty capped so log(Tz) always contributes meaningfully.
    """
    _EVAL = jax.jit(loss_fn)

    # State normalization
    xc = jnp.array((BOUNDS[:, 0] + BOUNDS[:, 1]) / 2.0)
    xs = jnp.array((BOUNDS[:, 1] - BOUNDS[:, 0]) / 2.0)
    TZ_MU, TZ_SC = float(jnp.log(jnp.array(50.0))), 3.0
    TX_MU, TX_SC = float(jnp.log(jnp.array(0.5))),  2.0

    def make_state(x_nom, Tz, Tx, Tz_prev, Tx_prev):
        log_Tz      = jnp.log(jnp.clip(jnp.array(Tz),      1e-3, 1e9))
        log_Tx      = jnp.log(jnp.clip(jnp.array(Tx),      1e-9, 1e3))
        log_Tz_prev = jnp.log(jnp.clip(jnp.array(Tz_prev), 1e-3, 1e9))
        log_Tx_prev = jnp.log(jnp.clip(jnp.array(Tx_prev), 1e-9, 1e3))
        return jnp.concatenate([
            (x_nom - xc) / xs,
            jnp.array([(log_Tz - TZ_MU) / TZ_SC,
                        (log_Tx - TX_MU) / TX_SC,
                        log_Tz - log_Tz_prev,   # positive → Tz improved this step
                        log_Tx - log_Tx_prev]),  # positive → Tx improved this step
        ])

    SD, AD = 8, 4   # 8-D state: x(4) + logTz + logTx + ΔlogTz + ΔlogTx

    # Networks
    rng = jax.random.PRNGKey(seed)
    rng, k1, k2 = jax.random.split(rng, 3)
    ap = _init_mlp(k1, SD, hidden_size, AD * 2)
    cp = _init_mlp(k2, SD, hidden_size, 1)
    za = jax.tree_util.tree_map(jnp.zeros_like, ap)
    zc = jax.tree_util.tree_map(jnp.zeros_like, cp)
    am, av = za, za
    cm, cv = zc, zc

    # JIT-compiled combined PPO loss
    # Fix 1: log_std clamped to (-3, -2) matching rollout → no distribution mismatch
    def _loss(ap_, cp_, states, actions, old_lp, advs, rets):
        out  = jax.vmap(lambda s: _mlp(ap_, s))(states)
        mu   = action_scale * jnp.tanh(out[:, :AD])
        lst  = jnp.clip(out[:, AD:], -3.0, -2.0)            # Fix 1: tight std
        std  = jnp.exp(lst)
        nlp  = -0.5 * jnp.sum(((actions - mu) / std)**2 + 2*lst + jnp.log(2*jnp.pi), axis=-1)
        r    = jnp.exp(jnp.clip(nlp - old_lp, -5.0, 5.0))
        adv_n = (advs - advs.mean()) / (advs.std() + 1e-8)
        surr  = jnp.minimum(r * adv_n, jnp.clip(r, 1-clip_eps, 1+clip_eps) * adv_n)
        ent   = jnp.mean(jnp.sum(lst + 0.5*jnp.log(2*jnp.pi*jnp.e), axis=-1))
        vals  = jax.vmap(lambda s: _mlp(cp_, s))(states).squeeze(-1)
        return -jnp.mean(surr) - entropy_coef*ent + 0.5*jnp.mean((vals - rets)**2)

    ppo_grad = jax.jit(jax.value_and_grad(_loss, argnums=(0, 1)))

    # Warm-up compile with correct (horizon, SD=8) shape
    _z = lambda sh: jnp.zeros(sh)
    ppo_grad(ap, cp, _z((horizon, SD)), _z((horizon, AD)),
             _z(horizon), _z(horizon), _z(horizon))

    # Rollout buffers
    bs, ba, blp, br, bv = (np.zeros((horizon, SD)), np.zeros((horizon, AD)),
                            np.zeros(horizon), np.zeros(horizon), np.zeros(horizon))
    idx = t_opt = 0

    x = jnp.array(x0, dtype=jnp.float64)
    _, Tx_now, Tz_now = _EVAL(x)
    Tz_prev, Tx_prev = float(Tz_now), float(Tx_now)   # Fix 3: track prev for delta
    param_h, Tx_h, Tz_h = [], [], []

    for drift_np in drift_traj:
        d = jnp.array(drift_np)

        for _ in range(optimizer_freq):
            s = make_state(x, float(Tz_now), float(Tx_now), Tz_prev, Tx_prev)

            out    = _mlp(ap, s)
            mu     = action_scale * jnp.tanh(out[:AD])
            lst    = jnp.clip(out[AD:], -3.0, -2.0)         # Fix 1: tight std (max 0.14)
            rng, k = jax.random.split(rng)
            noise  = jax.random.normal(k, (AD,))
            action = mu + jnp.exp(lst) * noise
            lp     = float(-0.5 * jnp.sum(noise**2 + 2*lst + jnp.log(2*jnp.pi)))

            x_new = jnp.clip(x + action, BOUNDS_LO, BOUNDS_HI)
            _, Tx_now, Tz_now = _EVAL(jnp.clip(x_new + d, BOUNDS_LO, BOUNDS_HI))

            # Fix 2: cap ratio penalty so log(Tz) always contributes to reward
            ratio_err = float(jnp.clip(
                jnp.abs(Tz_now / jnp.clip(Tx_now, 1e-9, None) - nu), 0.0, 20.0))
            reward = (float(jnp.log(jnp.clip(Tz_now, 1e-3, None)))
                      + float(jnp.log(jnp.clip(Tx_now, 1e-9, None)))
                      - lam * ratio_err)

            bs[idx] = np.array(s)
            ba[idx] = np.array(action)
            blp[idx] = lp
            br[idx] = reward
            bv[idx] = float(_mlp(cp, s)[0])
            Tz_prev, Tx_prev = float(Tz_now), float(Tx_now)  # Fix 3: update delta ref
            idx += 1
            x = x_new

            if idx == horizon:
                s_next     = make_state(x, float(Tz_now), float(Tx_now), Tz_prev, Tx_prev)
                nv         = float(_mlp(cp, s_next)[0])
                advs, rets = _gae(br, bv, nv, gamma, gae_lambda)
                sj, aj, lpj = jnp.array(bs), jnp.array(ba), jnp.array(blp)
                aaj, rj     = jnp.array(advs), jnp.array(rets)

                for _ in range(n_ppo_epochs):
                    t_opt += 1
                    _, (g_ap, g_cp) = ppo_grad(ap, cp, sj, aj, lpj, aaj, rj)
                    ap, am, av = _adam_pytree(ap, g_ap, am, av, t_opt, lr_actor)
                    cp, cm, cv = _adam_pytree(cp, g_cp, cm, cv, t_opt, lr_critic)
                idx = 0

        param_h.append(np.array(x))
        Tx_h.append(float(Tx_now))
        Tz_h.append(float(Tz_now))

    return np.array(Tx_h), np.array(Tz_h), np.array(param_h)


# ── Rolling mean helper ────────────────────────────────────────────────────────

def _sm(x, w=7):
    return x if w <= 1 else np.convolve(x, np.ones(w) / w, mode="same")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nu",           type=float, default=1000.0)
    parser.add_argument("--lam",          type=float, default=0.5)
    parser.add_argument("--n-steps",      type=int,   default=300)
    parser.add_argument("--adam-steps",   type=int,   default=120)
    parser.add_argument("--lr",           type=float, default=0.02)
    parser.add_argument("--drift-period", type=int,   default=DRIFT_PERIOD)
    parser.add_argument("--drift-seed",   type=int,   default=0)
    parser.add_argument("--opt-freq",     type=int,   default=3,
                        help="PPO optimizer steps per drift step")
    parser.add_argument("--horizon",      type=int,   default=64)
    parser.add_argument("--ppo-epochs",   type=int,   default=4)
    parser.add_argument("--lr-ppo",       type=float, default=3e-4)
    parser.add_argument("--smooth",       type=int,   default=7)
    parser.add_argument("--no-plot",      action="store_true")
    args = parser.parse_args()

    # GPU selection
    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    if gpus:
        jax.config.update("jax_default_device", gpus[-1])
        print(f"Running on GPU: {gpus[-1]}")
    else:
        print("No GPU — running on CPU")

    loss_fn = make_loss_fn(args.nu, args.lam)
    batched = jax.jit(jax.vmap(loss_fn))

    print("JIT warm-up …")
    batched(jnp.ones((2, 4)))[0].block_until_ready()
    print("Done.\n")

    x0 = np.array([2.0, 0.0, 8.0, 0.0])

    # ── Phase A: static Adam ─────────────────────────────────────────────────
    print(f"── Phase A: Static Adam ({args.adam_steps} steps) ──")
    x_static = run_static_adam(loss_fn, x0, n_steps=args.adam_steps, lr=args.lr)
    _, Tx_s0, Tz_s0 = loss_fn(jnp.array(x_static))
    print(f"   x = {x_static[:3]}  Tz={float(Tz_s0):.1f} µs  Tx={float(Tx_s0):.4f} µs\n")

    # ── Drift trajectory ─────────────────────────────────────────────────────
    print(f"── Drift: {args.n_steps} steps, period={args.drift_period} ──")
    drift = generate_drift(args.n_steps, seed=args.drift_seed, period=args.drift_period)
    slow  = sinusoidal_only(args.n_steps, period=args.drift_period)
    print(f"   g₂ Δ ∈ [{drift[:,0].min():.3f}, {drift[:,0].max():.3f}] MHz")
    print(f"   ε_d Δ ∈ [{drift[:,2].min():.3f}, {drift[:,2].max():.3f}] MHz\n")

    # ── Phase B: static under drift ──────────────────────────────────────────
    print("── Phase B: Static x under drift ──")
    Tx_static, Tz_static = evaluate_static(batched, x_static, drift)
    print(f"   Tz mean={Tz_static.mean():.1f}, min={Tz_static.min():.1f} µs\n")

    # ── Phase C: online PPO ──────────────────────────────────────────────────
    print(f"── Phase C: Online PPO "
          f"(opt_freq={args.opt_freq}, horizon={args.horizon}, "
          f"ppo_epochs={args.ppo_epochs}, lr={args.lr_ppo:.0e}) ──")
    Tx_ppo, Tz_ppo, params_ppo = run_online_ppo(
        loss_fn, x_static, drift,
        lr_actor=args.lr_ppo, lr_critic=args.lr_ppo * 3,
        optimizer_freq=args.opt_freq,
        horizon=args.horizon,
        n_ppo_epochs=args.ppo_epochs,
        nu=args.nu, lam=args.lam,
    )
    print(f"   Tz mean={Tz_ppo.mean():.1f}, min={Tz_ppo.min():.1f} µs")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'Method':<22}  {'Tz mean':>9}  {'Tz min':>9}  {'Tx mean':>9}")
    print("─" * 56)
    for name, Tz, Tx in [
        ("Static (frozen)",  Tz_static, Tx_static),
        ("PPO (online)",     Tz_ppo,    Tx_ppo),
    ]:
        print(f"{name:<22}  {Tz.mean():9.1f}  {Tz.min():9.1f}  {Tx.mean():9.4f}")
    gain = Tz_ppo.mean() / max(Tz_static.mean(), 1e-9)
    print(f"\n  PPO mean Tz gain over static: {gain:.2f}×")

    if args.no_plot:
        return

    st = np.arange(args.n_steps)
    W  = args.smooth

    # ── Fig 1: Drift model ───────────────────────────────────────────────────
    fig1, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig1.suptitle(
        f"Drift model — sinusoidal (period={args.drift_period}) + OU noise (τ={OU_TAU})",
        fontsize=11)
    for ax, col, amp, label, title in [
        (axes[0], 0, DRIFT_AMP_G2,  "Re(g₂)",  "g₂ drift"),
        (axes[1], 2, DRIFT_AMP_EPS, "Re(ε_d)", "ε_d drift"),
    ]:
        ax.plot(st, drift[:, col], lw=0.9, label="Total (sin + OU)")
        ax.plot(st, slow[:, col], "r--", lw=1.4, alpha=0.8,
                label=f"Sinusoidal (±{amp} MHz)")
        ax.fill_between(st, drift[:, col], slow[:, col], alpha=0.18, label="OU component")
        ax.axhline(0, color="k", lw=0.5)
        ax.set(xlabel="Step", ylabel=f"Δ {label} (MHz)", title=title)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig("cat_ppo_drift.png", dpi=150)
    print("\nSaved cat_ppo_drift.png")

    # ── Fig 2: Static vs PPO lifetimes ───────────────────────────────────────
    ratio_s   = Tz_static / np.clip(Tx_static, 1e-9, None)
    ratio_ppo = Tz_ppo    / np.clip(Tx_ppo,    1e-9, None)

    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle(
        "Phase B vs C — Static frozen parameters vs PPO online stabilization",
        fontsize=11)
    for ax, Ys, Yp, ylabel, title in [
        (axes[0], Tz_static, Tz_ppo,    "T_z (µs)",  "Bit-flip T_z"),
        (axes[1], Tx_static, Tx_ppo,    "T_x (µs)",  "Phase-flip T_x"),
        (axes[2], ratio_s,   ratio_ppo, "T_z / T_x", "Noise bias"),
    ]:
        ax.semilogy(st, _sm(Ys, W), color="C3", linestyle=":", lw=1.2,
                    alpha=0.7, label="Static (frozen)")
        ax.semilogy(st, _sm(Yp, W), color="C4", lw=1.8, label="PPO (online)")
        if "bias" in title:
            ax.axhline(args.nu, color="r", linestyle="--", lw=1.0,
                       label=f"target ν={args.nu:.0f}")
        ax.set(xlabel="Drift step", ylabel=ylabel, title=title)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("cat_ppo_baseline.png", dpi=150)
    print("Saved cat_ppo_baseline.png")

    # ── Fig 3: Effective parameters + tracking error ─────────────────────────
    x_eff_ppo    = np.clip(params_ppo + drift, BOUNDS[:, 0], BOUNDS[:, 1])
    x_eff_static = np.clip(x_static[None, :] + drift, BOUNDS[:, 0], BOUNDS[:, 1])
    ideal_g2     = np.full(args.n_steps, x_static[0])
    ideal_eps    = np.full(args.n_steps, x_static[2])

    fig3, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig3.suptitle(
        "Phase C — Effective parameters under PPO feedback\n"
        f"(effective = nominal set-point + hardware drift, "
        f"opt_freq={args.opt_freq}, horizon={args.horizon})",
        fontsize=11)

    ax = axes[0, 0]
    ax.plot(st, _sm(x_eff_ppo[:, 0],    W), color="C4", label="PPO effective g₂")
    ax.plot(st, _sm(x_eff_static[:, 0], W), "k:", lw=0.8, alpha=0.5,
            label="Static (no feedback)")
    ax.axhline(x_static[0], color="k", lw=1.2, linestyle="--",
               label=f"Ideal target ({x_static[0]:.3f})")
    ax.set(xlabel="Step", ylabel="Re(g₂) effective [MHz]",
           title="Effective g₂  (feedback keeps it near target)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(st, _sm(x_eff_ppo[:, 2],    W), color="C4", label="PPO effective ε_d")
    ax.plot(st, _sm(x_eff_static[:, 2], W), "k:", lw=0.8, alpha=0.5,
            label="Static (no feedback)")
    ax.axhline(x_static[2], color="k", lw=1.2, linestyle="--",
               label=f"Ideal target ({x_static[2]:.3f})")
    ax.set(xlabel="Step", ylabel="Re(ε_d) effective [MHz]",
           title="Effective ε_d  (feedback keeps it near target)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(st, _sm(x_eff_ppo[:, 0] - ideal_g2,    W), color="C4", label="PPO tracking error")
    ax.plot(st, _sm(x_eff_static[:, 0] - ideal_g2, W), "k:", lw=0.8, alpha=0.5,
            label="Static error (= drift)")
    ax.axhline(0, color="k", lw=1.0)
    ax.set(xlabel="Step", ylabel="Error in Re(g₂) [MHz]",
           title="g₂ tracking error  (0 = perfect)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(st, _sm(x_eff_ppo[:, 2] - ideal_eps,    W), color="C4", label="PPO tracking error")
    ax.plot(st, _sm(x_eff_static[:, 2] - ideal_eps, W), "k:", lw=0.8, alpha=0.5,
            label="Static error (= drift)")
    ax.axhline(0, color="k", lw=1.0)
    ax.set(xlabel="Step", ylabel="Error in Re(ε_d) [MHz]",
           title="ε_d tracking error  (0 = perfect)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3.savefig("cat_ppo_detail.png", dpi=150)
    print("Saved cat_ppo_detail.png")

    plt.show()


if __name__ == "__main__":
    main()
