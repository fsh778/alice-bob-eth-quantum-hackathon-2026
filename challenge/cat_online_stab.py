#!/usr/bin/env python3
"""
Cat qubit online stabilization under realistic parameter drift.

Three phases, each producing separate saved plots:
  A  Static optimization (no drift) — CMA-ES and Adam find good operating points.
     → cat_static_opt.png
  B  Freeze those points and evaluate under drift — shows lifetime degradation.
     → cat_drift_model.png, cat_static_drift.png
  C  Continuous online re-optimization under the same drift — shows stabilization.
     → cat_online_stab.png

Drift model applied to Re(g₂) and Re(ε_d):
  Sinusoidal: models slow environmental drift (temperature, flux bias, hours timescale).
              g₂ amp 0.25 MHz (~13%), ε_d amp 0.60 MHz (~8%), 150-step period.
  Ornstein-Uhlenbeck: correlated stochastic noise from control electronics.
              σ_g2=0.03 MHz, σ_eps=0.08 MHz, correlation length 30 steps.
  Both components use a fixed RNG seed, so all four methods see identical drift
  and comparisons are apples-to-apples.

Online stabilization mechanism:
  CMA-ES: each step draws a population around the current mean, evaluates every
          candidate at x_nominal + drift(t), tells the optimizer → mean tracks
          the drift-shifted optimum without any explicit drift model.
  Adam:   the gradient is computed at x_actual = x_nominal + drift(t).
          Because ∂x_actual/∂x_nominal = I, this gradient = ∂L/∂x_nominal and
          directly encodes how x_nominal must shift to compensate for drift.

Usage:
    python cat_online_stab.py
    python cat_online_stab.py --n-steps 400 --n-cmaes-epochs 40 --n-adam-steps 150
    python cat_online_stab.py --no-plot        # run numerics, skip figures
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from cmaes import SepCMA
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from cat_lifetime_opt import make_loss_fn

jax.config.update("jax_enable_x64", True)

# ── Drift model constants ──────────────────────────────────────────────────────
#
#  Inspired by the challenge notebook (pi-pulse example):
#    amp_factor_delta_drift = 0.8 * sin(2π·0.01·epoch)   (±80% drift over 100 epochs)
#
#  Scaled here to cat qubit parameters: g₂ ≈ 1.9 MHz, ε_d ≈ 7.5 MHz.
#  Sinusoidal amplitudes represent typical flux / amplitude calibration errors
#  in superconducting circuits.
DRIFT_PERIOD = 300  # steps per sinusoidal cycle (slow drift → optimizer can keep up)
DRIFT_AMP_G2 = 0.25  # sinusoidal amplitude for Re(g₂) [MHz]
DRIFT_AMP_EPS = 0.60  # sinusoidal amplitude for Re(ε_d) [MHz]
DRIFT_PHASE = np.pi / 3  # phase offset between g₂ and ε_d drifts

OU_TAU = 30  # Ornstein-Uhlenbeck correlation length [steps]
OU_STD_G2 = 0.030  # OU noise std for g₂ [MHz]
OU_STD_EPS = 0.080  # OU noise std for ε_d [MHz]

# ── Parameter bounds (same as sweep) ─────────────────────────────────────────
G2_MIN, G2_MAX = 1.2, 3.5
EPSD_MIN, EPSD_MAX = 2.0, 9.5
BOUNDS = np.array(
    [
        [G2_MIN, G2_MAX],
        [-0.5, 0.5],
        [EPSD_MIN, EPSD_MAX],
        [-0.5, 0.5],
    ]
)
BOUNDS_LO = jnp.array(BOUNDS[:, 0])
BOUNDS_HI = jnp.array(BOUNDS[:, 1])


# ── Drift trajectory generation ────────────────────────────────────────────────


def generate_drift_trajectory(
    n_steps: int, seed: int = 0, period: int = DRIFT_PERIOD
) -> np.ndarray:
    """
    Pre-generate a deterministic drift trajectory of shape (n_steps, 4).
    Layout: [Δg₂, 0, Δε_d, 0].

    Using a fixed seed and pre-generating ensures CMA-ES, Adam, static, and
    online methods all experience identical drift — necessary for fair comparison.
    The `period` argument overrides DRIFT_PERIOD for runtime tuning.
    """
    rng = np.random.default_rng(seed)
    ou_g2 = ou_eps = 0.0
    dt = 1.0 / OU_TAU
    traj = np.zeros((n_steps, 4))

    for i in range(n_steps):
        # Sinusoidal slow component (environmental)
        slow_g2 = DRIFT_AMP_G2 * np.sin(2.0 * np.pi * i / period)
        slow_eps = DRIFT_AMP_EPS * np.sin(2.0 * np.pi * i / period + DRIFT_PHASE)
        # OU stochastic component (control electronics)
        ou_g2 = ou_g2 * (1.0 - dt) + OU_STD_G2 * rng.standard_normal()
        ou_eps = ou_eps * (1.0 - dt) + OU_STD_EPS * rng.standard_normal()
        traj[i, 0] = slow_g2 + ou_g2
        traj[i, 2] = slow_eps + ou_eps

    return traj


def sinusoidal_only(n_steps: int, period: int = DRIFT_PERIOD) -> np.ndarray:
    """Pure sinusoidal drift (no OU) — used for overlay in drift plot."""
    t = np.arange(n_steps)
    traj = np.zeros((n_steps, 4))
    traj[:, 0] = DRIFT_AMP_G2 * np.sin(2.0 * np.pi * t / period)
    traj[:, 2] = DRIFT_AMP_EPS * np.sin(2.0 * np.pi * t / period + DRIFT_PHASE)
    return traj


# ── Phase A: static optimizers ────────────────────────────────────────────────


def run_static_cmaes(batched_loss, loss_fn, x0, n_epochs, batch_size):
    """Standard CMA-ES: no drift, returns best mean and convergence history."""
    opt = SepCMA(
        mean=np.array(x0, dtype=float),
        sigma=0.15,
        bounds=BOUNDS,
        population_size=batch_size,
        seed=42,
    )
    loss_h, Tx_h, Tz_h = [], [], []

    for ep in range(n_epochs):
        xs = jnp.array([opt.ask() for _ in range(batch_size)])
        losses, Txs, Tzs = batched_loss(xs)
        xs_np = np.array(xs)
        opt.tell([(xs_np[i], float(losses[i])) for i in range(batch_size)])

        loss_h.append(float(np.mean(losses)))
        Tx_h.append(float(np.mean(Txs)))
        Tz_h.append(float(np.mean(Tzs)))
        if ep % 10 == 0:
            print(
                f"    epoch {ep:3d}  loss={loss_h[-1]:.3f}  "
                f"Tz={Tz_h[-1]:.1f}  Tx={Tx_h[-1]:.4f}"
            )

    # Evaluate final mean
    _, Tx_f, Tz_f = loss_fn(jnp.array(opt.mean))
    print(
        f"    Final: g2={opt.mean[0]:.4f}, eps_d={opt.mean[2]:.4f}  "
        f"Tz={float(Tz_f):.1f} µs  Tx={float(Tx_f):.4f} µs"
    )
    return opt.mean.copy(), np.array(loss_h), np.array(Tx_h), np.array(Tz_h)


def run_static_adam(loss_fn, x0, n_steps, lr):
    """Standard Adam via JAX autodiff: no drift, returns best params and history."""

    def loss_with_aux(x):
        loss, Tx, Tz = loss_fn(x)
        return loss, (Tx, Tz)

    grad_fn = jax.jit(jax.value_and_grad(loss_with_aux, has_aux=True))
    x = jnp.array(x0, dtype=jnp.float64)
    m = v = jnp.zeros(4)
    loss_h, Tx_h, Tz_h = [], [], []

    for t in range(1, n_steps + 1):
        (lv, (Tv, Zv)), g = grad_fn(x)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g**2
        m_hat = m / (1.0 - 0.9**t)
        v_hat = v / (1.0 - 0.999**t)
        x = x - lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)
        x = jnp.clip(x, BOUNDS_LO, BOUNDS_HI)
        loss_h.append(float(lv))
        Tx_h.append(float(Tv))
        Tz_h.append(float(Zv))
        if t % 20 == 0:
            print(
                f"    step {t:4d}  loss={float(lv):.3f}  "
                f"Tz={float(Zv):.1f}  Tx={float(Tv):.4f}"
            )

    return np.array(x), np.array(loss_h), np.array(Tx_h), np.array(Tz_h)


# ── Phase B: static evaluation under drift ────────────────────────────────────


def evaluate_under_drift(
    batched_loss, x_fixed: np.ndarray, drift_traj: np.ndarray
) -> tuple:
    """
    Frozen-parameter evaluation across the full drift trajectory.

    Applies the precomputed drift to x_fixed at every step and batch-evaluates
    the entire trajectory in one vmap call — efficient because the drift is
    already known.
    """
    x_all = np.clip(x_fixed[None, :] + drift_traj, BOUNDS[:, 0], BOUNDS[:, 1])
    _, Txs, Tzs = batched_loss(jnp.array(x_all))
    return np.array(Txs), np.array(Tzs)


# ── Phase C: online CMA-ES ────────────────────────────────────────────────────


def run_online_cmaes(
    batched_loss,
    loss_fn,
    x0: np.ndarray,
    drift_traj: np.ndarray,
    batch_size: int,
    sigma: float = 0.08,
    optimizer_freq: int = 1,
) -> tuple:
    """
    Online CMA-ES with configurable time resolution.

    `optimizer_freq` population evaluations are executed per drift step,
    increasing the ratio of optimizer bandwidth to drift bandwidth.
    With optimizer_freq=3 and drift_period=300, the optimizer makes ~900
    updates per drift cycle versus only ~300 before — allowing it to track
    faster transients within each slow drift period.

    At each drift step:
      1. Draw optimizer_freq batches, each evaluated at x_nominal + drift(t).
      2. Tell CMA-ES after each batch → mean converges toward shifted optimum.
      3. Record mean (feedback set point) after all inner steps.
    """
    opt = SepCMA(
        mean=np.array(x0, dtype=float),
        sigma=sigma,
        bounds=BOUNDS,
        population_size=batch_size,
        seed=42,
    )
    mean_hist = []

    for drift_vec in drift_traj:
        for _ in range(optimizer_freq):
            xs_nom = np.array([opt.ask() for _ in range(batch_size)])
            xs_act = np.clip(xs_nom + drift_vec[None, :], BOUNDS[:, 0], BOUNDS[:, 1])
            losses, _, _ = batched_loss(jnp.array(xs_act))
            opt.tell([(xs_nom[i], float(losses[i])) for i in range(batch_size)])
        mean_hist.append(opt.mean.copy())

    # Batch-evaluate the nominal-mean trajectory at actual (drifted) operating points
    means = np.array(mean_hist)
    x_eval = np.clip(means + drift_traj, BOUNDS[:, 0], BOUNDS[:, 1])
    _, Txs, Tzs = batched_loss(jnp.array(x_eval))
    return np.array(Txs), np.array(Tzs), means


# ── Phase C: online Adam ──────────────────────────────────────────────────────


def run_online_adam(
    loss_fn,
    x0: np.ndarray,
    drift_traj: np.ndarray,
    lr: float,
    optimizer_freq: int = 1,
    batched_loss=None,
) -> tuple:
    """
    Online Adam with configurable time resolution.

    `optimizer_freq` gradient steps per drift step increase optimizer bandwidth.
    The global Adam step counter t_global ensures bias-correction stays correct
    across all inner iterations — bias correction (1 - β^t) converges smoothly.

    At x_actual = x_nominal + drift(t), gradient ∂L/∂x_nominal = ∂L/∂x_actual
    directly encodes the direction x_nominal must move to restore performance.
    """

    def loss_with_drift(x_nom, d):
        x_act = jnp.clip(x_nom + d, BOUNDS_LO, BOUNDS_HI)
        loss, Tx, Tz = loss_fn(x_act)
        return loss, (Tx, Tz)

    grad_fn = jax.jit(jax.value_and_grad(loss_with_drift, argnums=0, has_aux=True))

    x = jnp.array(x0, dtype=jnp.float64)
    m = v = jnp.zeros(4)
    t_global = 0  # global step counter: ensures correct bias correction
    param_hist = []

    for drift_np in drift_traj:
        d = jnp.array(drift_np)
        for _ in range(optimizer_freq):
            t_global += 1
            (_, _), g = grad_fn(x, d)
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g**2
            m_hat = m / (1.0 - 0.9**t_global)
            v_hat = v / (1.0 - 0.999**t_global)
            x = x - lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)
            x = jnp.clip(x, BOUNDS_LO, BOUNDS_HI)
        param_hist.append(np.array(x))

    params = np.array(param_hist)
    x_eval = np.clip(params + drift_traj, BOUNDS[:, 0], BOUNDS[:, 1])
    if batched_loss is not None:
        _, Txs, Tzs = batched_loss(jnp.array(x_eval))
    else:
        _, Txs, Tzs = jax.jit(jax.vmap(loss_fn))(jnp.array(x_eval))
    return np.array(Txs), np.array(Tzs), params


# ── Phase D: predictive online methods ───────────────────────────────────────


def run_online_adam_predictive(
    loss_fn,
    x0: np.ndarray,
    drift_traj: np.ndarray,
    lr: float,
    optimizer_freq: int = 1,
    lookahead: float = 0.5,
    batched_loss=None,
) -> tuple:
    """
    Predictive Adam: gradient lookahead for proactive drift compensation.

    After all optimizer_freq gradient steps, an additional fraction `lookahead`
    of the current Adam step is applied in the direction of the running gradient
    momentum. Because drift is correlated (OU + sinusoidal), the next gradient
    will be similar to the current one — applying a fraction of it now
    pre-positions the parameters before the next drift step arrives.

    Formally: x_pred = x - lookahead × lr × m̂ / (√v̂ + ε)
    This is a Nesterov-like lookahead but adapted to the online drift-tracking
    setting rather than convex acceleration.

    Proposed further improvement: replace the fixed lookahead fraction with a
    Kalman-gain-weighted predictor that adapts to estimated drift velocity.
    At low drift rate (near sinusoidal peak) the gain should shrink; at high
    rate (zero-crossing) it should grow. A 2-state OU Kalman filter tracking
    [drift_level, drift_velocity] could implement this with ~10 lines of numpy.
    """

    def loss_with_drift(x_nom, d):
        x_act = jnp.clip(x_nom + d, BOUNDS_LO, BOUNDS_HI)
        loss, Tx, Tz = loss_fn(x_act)
        return loss, (Tx, Tz)

    grad_fn = jax.jit(jax.value_and_grad(loss_with_drift, argnums=0, has_aux=True))

    x = jnp.array(x0, dtype=jnp.float64)
    m = v = jnp.zeros(4)
    t_global = 0
    param_hist = []

    for drift_np in drift_traj:
        d = jnp.array(drift_np)
        last_m_hat = last_v_hat = None

        for _ in range(optimizer_freq):
            t_global += 1
            (_, _), g = grad_fn(x, d)
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g**2
            last_m_hat = m / (1.0 - 0.9**t_global)
            last_v_hat = v / (1.0 - 0.999**t_global)
            x = x - lr * last_m_hat / (jnp.sqrt(last_v_hat) + 1e-8)
            x = jnp.clip(x, BOUNDS_LO, BOUNDS_HI)

        # Lookahead: apply fraction of current momentum step as predictive correction
        if last_m_hat is not None:
            x = x - lookahead * lr * last_m_hat / (jnp.sqrt(last_v_hat) + 1e-8)
            x = jnp.clip(x, BOUNDS_LO, BOUNDS_HI)

        param_hist.append(np.array(x))

    params = np.array(param_hist)
    x_eval = np.clip(params + drift_traj, BOUNDS[:, 0], BOUNDS[:, 1])
    if batched_loss is not None:
        _, Txs, Tzs = batched_loss(jnp.array(x_eval))
    else:
        _, Txs, Tzs = jax.jit(jax.vmap(loss_fn))(jnp.array(x_eval))
    return np.array(Txs), np.array(Tzs), params


def run_online_cmaes_predictive(
    batched_loss,
    loss_fn,
    x0: np.ndarray,
    drift_traj: np.ndarray,
    batch_size: int,
    sigma: float = 0.08,
    optimizer_freq: int = 1,
    lookahead: float = 0.5,
) -> tuple:
    """
    Predictive CMA-ES: mean momentum extrapolation.

    After all optimizer_freq evaluations, the mean's velocity between consecutive
    drift steps is computed and used to extrapolate: the set point is advanced by
    `lookahead × velocity` in anticipation of the continued drift. Because the
    drift is smooth (slow sinusoidal + correlated OU), the mean's movement is a
    reliable proxy for the drift direction.

    mean_predicted = mean + lookahead × (mean_current − mean_previous)

    The extrapolated mean is used as the set-point fed back to hardware.
    The internal CMA-ES state (its own mean, covariance) is NOT modified —
    only the recorded feedback signal carries the prediction. This preserves
    CMA-ES exploration while still benefiting from trend extrapolation.

    Proposed further improvement: use a receding-horizon prediction over the
    last K mean velocities (linear regression) to detect and track trend changes,
    especially around the sinusoidal drift zero-crossings where the direction
    reverses and a simple 1-step extrapolation would overshoot.
    """
    opt = SepCMA(
        mean=np.array(x0, dtype=float),
        sigma=sigma,
        bounds=BOUNDS,
        population_size=batch_size,
        seed=42,
    )
    mean_hist = []

    for drift_vec in drift_traj:
        mean_before = opt.mean.copy()
        for _ in range(optimizer_freq):
            xs_nom = np.array([opt.ask() for _ in range(batch_size)])
            xs_act = np.clip(xs_nom + drift_vec[None, :], BOUNDS[:, 0], BOUNDS[:, 1])
            losses, _, _ = batched_loss(jnp.array(xs_act))
            opt.tell([(xs_nom[i], float(losses[i])) for i in range(batch_size)])

        # Mean momentum extrapolation: advance in the direction the mean has moved
        velocity = opt.mean - mean_before
        mean_pred = np.clip(opt.mean + lookahead * velocity, BOUNDS[:, 0], BOUNDS[:, 1])
        mean_hist.append(mean_pred)

    means = np.array(mean_hist)
    x_eval = np.clip(means + drift_traj, BOUNDS[:, 0], BOUNDS[:, 1])
    _, Txs, Tzs = batched_loss(jnp.array(x_eval))
    return np.array(Txs), np.array(Tzs), means


# ── Plotting helpers ──────────────────────────────────────────────────────────


def _smth(x, w=5):
    """Uniform rolling mean for visual clarity (does not affect data)."""
    if w <= 1:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


# ── PPO helper functions ───────────────────────────────────────────────────────
# Pure-JAX MLP + Adam-on-pytrees.  No dependencies beyond jax and numpy.


def _ppo_init_mlp(rng_key, in_dim: int, hidden: int, out_dim: int):
    """He-initialised 2-hidden-layer MLP.  Returns list of (W, b) tuples."""
    params = []
    for fan_in, fan_out in [(in_dim, hidden), (hidden, hidden), (hidden, out_dim)]:
        rng_key, sub = jax.random.split(rng_key)
        W = jax.random.normal(sub, (fan_in, fan_out)) * jnp.sqrt(2.0 / fan_in)
        b = jnp.zeros(fan_out)
        params.append((W, b))
    return params


def _ppo_mlp(params, x):
    """Forward pass: tanh on hidden layers, linear output."""
    for i, (W, b) in enumerate(params):
        x = x @ W + b
        if i < len(params) - 1:
            x = jnp.tanh(x)
    return x


def _ppo_adam_update(params, grads, m, v, t, lr, b1=0.9, b2=0.999, eps=1e-8):
    """Adam update on any JAX pytree (works with list-of-tuples MLP params)."""
    m = jax.tree_util.tree_map(lambda mi, gi: b1 * mi + (1 - b1) * gi, m, grads)
    v = jax.tree_util.tree_map(lambda vi, gi: b2 * vi + (1 - b2) * gi**2, v, grads)
    m_hat = jax.tree_util.tree_map(lambda mi: mi / (1 - b1**t), m)
    v_hat = jax.tree_util.tree_map(lambda vi: vi / (1 - b2**t), v)
    params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
        params,
        m_hat,
        v_hat,
    )
    return params, m, v


def _ppo_gae(
    rewards: np.ndarray, values: np.ndarray, next_value: float, gamma: float, lam: float
):
    """GAE advantages and bootstrapped returns, shape (T,) each."""
    T = len(rewards)
    adv = np.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        nv = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * nv - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    return adv, adv + values


# ── Phase E: online PPO ────────────────────────────────────────────────────────


def run_online_ppo(
    loss_fn,
    x0: np.ndarray,
    drift_traj: np.ndarray,
    lr_actor: float = 3e-4,
    lr_critic: float = 1e-3,
    optimizer_freq: int = 1,
    horizon: int = 64,
    n_ppo_epochs: int = 4,
    clip_eps: float = 0.2,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    entropy_coef: float = 0.01,
    action_scale: float = 0.25,
    hidden_size: int = 64,
    seed: int = 7,
    batched_loss=None,
    nu: float = 1000.0,
    lam: float = 0.5,
) -> tuple:
    """
    Online PPO optimizer for cat-qubit parameter stabilization.

    Three bugs fixed vs the original 7-D / uncapped implementation:

    Fix 1 — log_std tightened (-4,-0.5) → (-3,-2): max std 0.61 → 0.14.
      The old range let noise swamp the tanh-bounded mean, producing actions
      far outside the intended ±action_scale range and creating a distribution
      mismatch in the PPO importance ratio that the clip couldn't correct.

    Fix 2 — bias penalty capped at 20 in the reward.
      Uncapped |Tz/Tx − ν| during exploration can be thousands, completely
      drowning log(Tz) (≲ 10).  The policy optimised ratio satisfaction at
      any cost, including shrinking α so Tx grew while Tz collapsed.

    Fix 3 — state trend signals replace slow reward EMA.
      Δlog Tz and Δlog Tx give the policy direct per-step feedback on whether
      lifetimes are improving, enabling drift-direction inference without
      observing the drift vector directly.

    State (8-D)
    -----------
    • Normalized nominal params  x̂  (4-D)
    • log T_z  normalized              (1-D)
    • log T_x  normalized              (1-D)
    • Δ log T_z  (current − previous)  (1-D)   ← Fix 3
    • Δ log T_x  (current − previous)  (1-D)   ← Fix 3

    Reward
    ------
    log(Tz) + log(Tx) − lam × clip(|Tz/Tx − ν|, 0, 20)   ← Fix 2
    """
    _EVAL = jax.jit(loss_fn)

    # ── State normalization constants ────────────────────────────────────────
    _x_c = jnp.array((BOUNDS[:, 0] + BOUNDS[:, 1]) / 2.0)
    _x_s = jnp.array((BOUNDS[:, 1] - BOUNDS[:, 0]) / 2.0)
    _TZ_LOG_MU = float(jnp.log(jnp.array(50.0)))
    _TZ_LOG_SCALE = 3.0
    _TX_LOG_MU = float(jnp.log(jnp.array(0.5)))
    _TX_LOG_SCALE = 2.0

    def make_state(x_nom, Tz, Tx, Tz_prev, Tx_prev):
        log_Tz = jnp.log(jnp.clip(jnp.array(Tz), 1e-3, 1e9))
        log_Tx = jnp.log(jnp.clip(jnp.array(Tx), 1e-9, 1e3))
        log_Tz_prev = jnp.log(jnp.clip(jnp.array(Tz_prev), 1e-3, 1e9))
        log_Tx_prev = jnp.log(jnp.clip(jnp.array(Tx_prev), 1e-9, 1e3))
        x_n = (x_nom - _x_c) / _x_s
        return jnp.concatenate(
            [
                x_n,
                jnp.array(
                    [
                        (log_Tz - _TZ_LOG_MU) / _TZ_LOG_SCALE,
                        (log_Tx - _TX_LOG_MU) / _TX_LOG_SCALE,
                        log_Tz - log_Tz_prev,  # positive → Tz improved this step
                        log_Tx - log_Tx_prev,  # positive → Tx improved this step
                    ]
                ),
            ]
        )

    STATE_DIM = 8  # x(4) + logTz + logTx + ΔlogTz + ΔlogTx
    ACTION_DIM = 4
    ACTOR_OUT = ACTION_DIM * 2

    # ── Network and Adam-state initialisation ────────────────────────────────
    rng = jax.random.PRNGKey(seed)
    rng, k1, k2 = jax.random.split(rng, 3)
    ap = _ppo_init_mlp(k1, STATE_DIM, hidden_size, ACTOR_OUT)
    cp = _ppo_init_mlp(k2, STATE_DIM, hidden_size, 1)

    zero_a = jax.tree_util.tree_map(jnp.zeros_like, ap)
    zero_c = jax.tree_util.tree_map(jnp.zeros_like, cp)
    am, av = zero_a, zero_a
    cm, cv = zero_c, zero_c

    # ── JIT-compiled combined PPO loss + gradient ────────────────────────────
    def _ppo_loss(ap_, cp_, states, actions, old_lp, advs, rets):
        out = jax.vmap(lambda s: _ppo_mlp(ap_, s))(states)
        mu = action_scale * jnp.tanh(out[:, :ACTION_DIM])
        lst = jnp.clip(out[:, ACTION_DIM:], -3.0, -2.0)  # Fix 1
        std = jnp.exp(lst)
        nlp = -0.5 * jnp.sum(
            ((actions - mu) / std) ** 2 + 2 * lst + jnp.log(2 * jnp.pi), axis=-1
        )
        r = jnp.exp(jnp.clip(nlp - old_lp, -5.0, 5.0))
        adv_n = (advs - advs.mean()) / (advs.std() + 1e-8)
        surr = jnp.minimum(
            r * adv_n, jnp.clip(r, 1.0 - clip_eps, 1.0 + clip_eps) * adv_n
        )
        a_loss = -jnp.mean(surr)
        entropy = jnp.mean(jnp.sum(lst + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1))
        vals = jax.vmap(lambda s: _ppo_mlp(cp_, s))(states).squeeze(-1)
        c_loss = jnp.mean((vals - rets) ** 2)
        return a_loss - entropy_coef * entropy + 0.5 * c_loss, (a_loss, c_loss, entropy)

    _ppo_grad = jax.jit(jax.value_and_grad(_ppo_loss, argnums=(0, 1), has_aux=True))

    # Warm-up: compile for the exact (horizon, STATE_DIM=8) shape
    _z = lambda shape: jnp.zeros(shape)
    _ppo_grad(
        ap,
        cp,
        _z((horizon, STATE_DIM)),
        _z((horizon, ACTION_DIM)),
        _z(horizon),
        _z(horizon),
        _z(horizon),
    )

    # ── Rollout buffers ──────────────────────────────────────────────────────
    buf_s = np.zeros((horizon, STATE_DIM))
    buf_a = np.zeros((horizon, ACTION_DIM))
    buf_lp = np.zeros(horizon)
    buf_r = np.zeros(horizon)
    buf_v = np.zeros(horizon)
    idx = 0
    t_opt = 0

    # ── Initial evaluation ───────────────────────────────────────────────────
    x = jnp.array(x0, dtype=jnp.float64)
    _, Tx_now, Tz_now = _EVAL(x)
    Tz_prev, Tx_prev = float(Tz_now), float(Tx_now)  # Fix 3: delta reference

    param_hist, Tx_hist, Tz_hist = [], [], []

    for drift_np in drift_traj:
        d = jnp.array(drift_np)

        for _ in range(optimizer_freq):
            state = make_state(x, float(Tz_now), float(Tx_now), Tz_prev, Tx_prev)

            # Actor forward ──────────────────────────────────────────────────
            out = _ppo_mlp(ap, state)
            mu = action_scale * jnp.tanh(out[:ACTION_DIM])
            lst = jnp.clip(out[ACTION_DIM:], -3.0, -2.0)  # Fix 1
            rng, k = jax.random.split(rng)
            noise = jax.random.normal(k, (ACTION_DIM,))
            action = mu + jnp.exp(lst) * noise
            log_prob = float(-0.5 * jnp.sum(noise**2 + 2 * lst + jnp.log(2 * jnp.pi)))

            # Environment step ───────────────────────────────────────────────
            x_new = jnp.clip(x + action, BOUNDS_LO, BOUNDS_HI)
            x_act = jnp.clip(x_new + d, BOUNDS_LO, BOUNDS_HI)
            _, Tx_now, Tz_now = _EVAL(x_act)

            # Fix 2: cap ratio penalty so log(Tz) always matters
            ratio_err = float(
                jnp.clip(jnp.abs(Tz_now / jnp.clip(Tx_now, 1e-9, None) - nu), 0.0, 20.0)
            )
            reward = (
                float(jnp.log(jnp.clip(Tz_now, 1e-3, None)))
                + float(jnp.log(jnp.clip(Tx_now, 1e-9, None)))
                - lam * ratio_err
            )
            value = float(_ppo_mlp(cp, state)[0])

            # Buffer store ───────────────────────────────────────────────────
            buf_s[idx] = np.array(state)
            buf_a[idx] = np.array(action)
            buf_lp[idx] = log_prob
            buf_r[idx] = reward
            buf_v[idx] = value
            Tz_prev, Tx_prev = float(Tz_now), float(Tx_now)  # Fix 3: update delta ref
            idx += 1
            x = x_new

            # PPO update when buffer full ─────────────────────────────────────
            if idx == horizon:
                s_next = make_state(x, float(Tz_now), float(Tx_now), Tz_prev, Tx_prev)
                nv = float(_ppo_mlp(cp, s_next)[0])
                advs, rets = _ppo_gae(buf_r, buf_v, nv, gamma, gae_lambda)

                sj = jnp.array(buf_s)
                aj = jnp.array(buf_a)
                lpj = jnp.array(buf_lp)
                aaj = jnp.array(advs)
                rj = jnp.array(rets)

                for _ in range(n_ppo_epochs):
                    t_opt += 1
                    (_, _), (g_ap, g_cp) = _ppo_grad(ap, cp, sj, aj, lpj, aaj, rj)
                    ap, am, av = _ppo_adam_update(ap, g_ap, am, av, t_opt, lr_actor)
                    cp, cm, cv = _ppo_adam_update(cp, g_cp, cm, cv, t_opt, lr_critic)
                idx = 0

        param_hist.append(np.array(x))
        Tx_hist.append(float(Tx_now))
        Tz_hist.append(float(Tz_now))

    return np.array(Tx_hist), np.array(Tz_hist), np.array(param_hist)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--nu", type=float, default=1000.0, help="Target T_z/T_x bias")
    parser.add_argument("--lam", type=float, default=0.5, help="Bias penalty weight")
    parser.add_argument(
        "--n-steps", type=int, default=300, help="Online drift steps (Phase B & C)"
    )
    parser.add_argument(
        "--n-cmaes-epochs", type=int, default=30, help="Static CMA-ES epochs (Phase A)"
    )
    parser.add_argument(
        "--n-adam-steps", type=int, default=100, help="Static Adam steps (Phase A)"
    )
    parser.add_argument("--batch", type=int, default=8, help="CMA-ES population size")
    parser.add_argument("--lr", type=float, default=0.02, help="Adam learning rate")
    parser.add_argument(
        "--drift-seed", type=int, default=0, help="RNG seed for drift trajectory"
    )
    parser.add_argument(
        "--drift-period",
        type=int,
        default=DRIFT_PERIOD,
        help="Sinusoidal drift period in steps (larger = slower drift)",
    )
    parser.add_argument(
        "--optimizer-freq",
        type=int,
        default=3,
        help="Optimizer steps per drift step (higher = faster feedback)",
    )
    parser.add_argument(
        "--lookahead",
        type=float,
        default=0.5,
        help="Predictive lookahead fraction for Phase D methods",
    )
    parser.add_argument(
        "--smooth", type=int, default=7, help="Rolling-mean window for plots (1=off)"
    )
    parser.add_argument(
        "--lr-ppo",
        type=float,
        default=3e-4,
        help="PPO actor learning rate (critic uses 3×)",
    )
    parser.add_argument(
        "--ppo-horizon",
        type=int,
        default=64,
        help="PPO rollout horizon (transitions before each update)",
    )
    parser.add_argument(
        "--ppo-epochs", type=int, default=4, help="PPO gradient epochs per rollout"
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip all figures")
    args = parser.parse_args()

    # ── Device setup ─────────────────────────────────────────────────────────
    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    if len(gpus) > 1:
        jax.config.update("jax_default_device", gpus[1])
        print(f"Running on GPU: {gpus[1]}  (gpu[0] reserved)")
    elif len(gpus) == 1:
        jax.config.update("jax_default_device", gpus[0])
        print(f"Running on GPU: {gpus[0]}")
    else:
        print("No GPU — running on CPU")

    # ── Build shared loss functions ───────────────────────────────────────────
    loss_fn = make_loss_fn(args.nu, args.lam)
    batched = jax.jit(jax.vmap(loss_fn))

    print("JIT warm-up …")
    _w = batched(jnp.ones((2, 4)))
    _w[0].block_until_ready()
    print("Done.\n")

    # ── Phase A: static optimization (no drift) ───────────────────────────────
    x0 = np.array(
        [2.0, 0.0, 8.0, 0.0]
    )  # headroom above stabilisation threshold (g2 ≈ 1.58)

    print(
        f"── Phase A: Static CMA-ES ({args.n_cmaes_epochs} epochs, "
        f"batch={args.batch}) ──"
    )
    x_cmaes, cmaes_loss, cmaes_Tx, cmaes_Tz = run_static_cmaes(
        batched, loss_fn, x0, args.n_cmaes_epochs, args.batch
    )

    print(f"\n── Phase A: Static Adam ({args.n_adam_steps} steps, lr={args.lr}) ──")
    x_adam, adam_loss, adam_Tx, adam_Tz = run_static_adam(
        loss_fn, x0, args.n_adam_steps, args.lr
    )
    print(
        f"    Final: g2={x_adam[0]:.4f}, eps_d={x_adam[2]:.4f}  "
        f"Tz={adam_Tz[-1]:.1f} µs  Tx={adam_Tx[-1]:.4f} µs"
    )

    # ── Drift trajectory ──────────────────────────────────────────────────────
    print(
        f"\n── Generating drift trajectory "
        f"({args.n_steps} steps, seed={args.drift_seed}) ──"
    )
    drift_traj = generate_drift_trajectory(
        args.n_steps, seed=args.drift_seed, period=args.drift_period
    )
    slow_traj = sinusoidal_only(args.n_steps, period=args.drift_period)
    print(f"   Period={args.drift_period} steps  optimizer_freq={args.optimizer_freq}")
    print(
        f"   Re(g₂):  [{drift_traj[:, 0].min():.3f}, {drift_traj[:, 0].max():.3f}] MHz"
    )
    print(
        f"   Re(ε_d): [{drift_traj[:, 2].min():.3f}, {drift_traj[:, 2].max():.3f}] MHz"
    )

    # ── Phase B: static evaluation under drift ────────────────────────────────
    print("\n── Phase B: Static parameters under drift ──")
    print("   CMA-ES static …")
    Tx_sc, Tz_sc = evaluate_under_drift(batched, x_cmaes, drift_traj)
    print(f"   T_z: mean={Tz_sc.mean():.1f}, min={Tz_sc.min():.1f} µs")
    print("   Adam static …")
    Tx_sa, Tz_sa = evaluate_under_drift(batched, x_adam, drift_traj)
    print(f"   T_z: mean={Tz_sa.mean():.1f}, min={Tz_sa.min():.1f} µs")

    # ── Phase C: online stabilization ────────────────────────────────────────
    print(f"\n── Phase C: Online CMA-ES ({args.n_steps} steps, batch={args.batch}) ──")
    Tx_oc, Tz_oc, params_oc = run_online_cmaes(
        batched,
        loss_fn,
        x_cmaes,
        drift_traj,
        args.batch,
        optimizer_freq=args.optimizer_freq,
    )
    print(f"   T_z: mean={Tz_oc.mean():.1f}, min={Tz_oc.min():.1f} µs")

    print(
        f"\n── Phase C: Online Adam ({args.n_steps} steps, lr={args.lr}, "
        f"freq={args.optimizer_freq}) ──"
    )
    Tx_oa, Tz_oa, params_oa = run_online_adam(
        loss_fn,
        x_adam,
        drift_traj,
        args.lr,
        optimizer_freq=args.optimizer_freq,
        batched_loss=batched,
    )
    print(f"   T_z: mean={Tz_oa.mean():.1f}, min={Tz_oa.min():.1f} µs")

    # ── Phase D: predictive online stabilization ──────────────────────────────
    print(
        f"\n── Phase D: Predictive CMA-ES ({args.n_steps} steps, "
        f"freq={args.optimizer_freq}, lookahead={args.lookahead}) ──"
    )
    Tx_pc, Tz_pc, params_pc = run_online_cmaes_predictive(
        batched,
        loss_fn,
        x_cmaes,
        drift_traj,
        args.batch,
        optimizer_freq=args.optimizer_freq,
        lookahead=args.lookahead,
    )
    print(f"   T_z: mean={Tz_pc.mean():.1f}, min={Tz_pc.min():.1f} µs")

    print(
        f"\n── Phase D: Predictive Adam ({args.n_steps} steps, "
        f"lr={args.lr}, freq={args.optimizer_freq}, lookahead={args.lookahead}) ──"
    )
    Tx_pa, Tz_pa, params_pa = run_online_adam_predictive(
        loss_fn,
        x_adam,
        drift_traj,
        args.lr,
        optimizer_freq=args.optimizer_freq,
        lookahead=args.lookahead,
        batched_loss=batched,
    )
    print(f"   T_z: mean={Tz_pa.mean():.1f}, min={Tz_pa.min():.1f} µs")

    print(
        f"\n── Phase E: Online PPO ({args.n_steps} steps, "
        f"lr={args.lr_ppo:.0e}, freq={args.optimizer_freq}, "
        f"horizon={args.ppo_horizon}, epochs={args.ppo_epochs}) ──"
    )
    Tx_ppo, Tz_ppo, params_ppo = run_online_ppo(
        loss_fn,
        x_adam,
        drift_traj,
        lr_actor=args.lr_ppo,
        lr_critic=args.lr_ppo * 3,
        optimizer_freq=args.optimizer_freq,
        horizon=args.ppo_horizon,
        n_ppo_epochs=args.ppo_epochs,
        batched_loss=batched,
        nu=args.nu,
        lam=args.lam,
    )
    print(f"   T_z: mean={Tz_ppo.mean():.1f}, min={Tz_ppo.min():.1f} µs")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Summary ──")
    hdr = f"{'Method':<26}  {'T_z mean':>10}  {'T_z min':>10}  {'T_x mean':>10}"
    print(hdr)
    print("─" * len(hdr))
    for name, Tz, Tx in [
        ("CMA-ES (static)", Tz_sc, Tx_sc),
        ("Adam   (static)", Tz_sa, Tx_sa),
        ("CMA-ES (online)", Tz_oc, Tx_oc),
        ("Adam   (online)", Tz_oa, Tx_oa),
        ("CMA-ES (predictive)", Tz_pc, Tx_pc),
        ("Adam   (predictive)", Tz_pa, Tx_pa),
        ("PPO    (online)", Tz_ppo, Tx_ppo),
    ]:
        print(f"{name:<26}  {Tz.mean():10.1f}  {Tz.min():10.1f}  {Tx.mean():10.4f}")

    if args.no_plot:
        return

    # ═════════════════════════════════════════════════════════════════════════
    # Plots
    # ═════════════════════════════════════════════════════════════════════════
    W = args.smooth
    ep_c = np.arange(args.n_cmaes_epochs)
    ep_a = np.arange(args.n_adam_steps)
    st = np.arange(args.n_steps)

    # ── Fig 1: Phase A — static optimization convergence ─────────────────────
    fig1, axes1 = plt.subplots(1, 3, figsize=(14, 4))
    fig1.suptitle(
        "Phase A — Static optimization (no drift)\n"
        f"CMA-ES: {args.n_cmaes_epochs} epochs, batch={args.batch}  |  "
        f"Adam: {args.n_adam_steps} steps, lr={args.lr}",
        fontsize=11,
    )

    axes1[0].plot(ep_c, cmaes_loss, label="CMA-ES")
    axes1[0].plot(ep_a, adam_loss, label="Adam (gradient)", linestyle="--")
    axes1[0].set(xlabel="Step / Epoch", ylabel="Loss", title="Loss convergence")
    axes1[0].legend()
    axes1[0].grid(True, alpha=0.3)

    axes1[1].semilogy(ep_c, cmaes_Tz, label="CMA-ES")
    axes1[1].semilogy(ep_a, adam_Tz, label="Adam", linestyle="--")
    axes1[1].set(xlabel="Step / Epoch", ylabel="T_z (µs)", title="Bit-flip T_z")
    axes1[1].legend()
    axes1[1].grid(True, alpha=0.3)

    axes1[2].semilogy(ep_c, cmaes_Tx, label="CMA-ES")
    axes1[2].semilogy(ep_a, adam_Tx, label="Adam", linestyle="--")
    axes1[2].set(xlabel="Step / Epoch", ylabel="T_x (µs)", title="Phase-flip T_x")
    axes1[2].legend()
    axes1[2].grid(True, alpha=0.3)

    fig1.tight_layout()
    fig1.savefig("cat_static_opt.png", dpi=150)
    print("\nSaved cat_static_opt.png")

    # ── Fig 2: Drift model visualisation ─────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4))
    fig2.suptitle(
        f"Drift model: sinusoidal (period={args.drift_period} steps) "
        f"+ Ornstein-Uhlenbeck noise (τ={OU_TAU} steps)",
        fontsize=11,
    )

    ax = axes2[0]
    ax.plot(st, drift_traj[:, 0], lw=0.9, color="C0", label="Total (sinusoidal + OU)")
    ax.plot(
        st,
        slow_traj[:, 0],
        "r--",
        lw=1.5,
        alpha=0.8,
        label=f"Sinusoidal (±{DRIFT_AMP_G2} MHz)",
    )
    ax.fill_between(
        st,
        drift_traj[:, 0],
        slow_traj[:, 0],
        alpha=0.18,
        color="C0",
        label="OU noise component",
    )
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="Drift step", ylabel="Δ Re(g₂) (MHz)", title="g₂ drift")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes2[1]
    ax.plot(st, drift_traj[:, 2], lw=0.9, color="C1", label="Total (sinusoidal + OU)")
    ax.plot(
        st,
        slow_traj[:, 2],
        "r--",
        lw=1.5,
        alpha=0.8,
        label=f"Sinusoidal (±{DRIFT_AMP_EPS} MHz)",
    )
    ax.fill_between(
        st,
        drift_traj[:, 2],
        slow_traj[:, 2],
        alpha=0.18,
        color="C1",
        label="OU noise component",
    )
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="Drift step", ylabel="Δ Re(ε_d) (MHz)", title="ε_d drift")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig("cat_drift_model.png", dpi=150)
    print("Saved cat_drift_model.png")

    # ── Fig 3: Phase B — static methods under drift ───────────────────────────
    ratio_sc = Tz_sc / np.clip(Tx_sc, 1e-9, None)
    ratio_sa = Tz_sa / np.clip(Tx_sa, 1e-9, None)

    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
    fig3.suptitle(
        "Phase B — Static parameters under drift  "
        "(parameters frozen at no-drift optimum)",
        fontsize=11,
    )

    ax = axes3[0]
    ax.semilogy(st, _smth(Tz_sc, W), label="CMA-ES", alpha=0.85)
    ax.semilogy(st, _smth(Tz_sa, W), label="Adam", linestyle="--", alpha=0.85)
    ax.set(
        xlabel="Drift step",
        ylabel="T_z (µs)",
        title="Bit-flip lifetime T_z  (degradation visible)",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes3[1]
    ax.semilogy(st, _smth(Tx_sc, W), label="CMA-ES", alpha=0.85)
    ax.semilogy(st, _smth(Tx_sa, W), label="Adam", linestyle="--", alpha=0.85)
    ax.set(xlabel="Drift step", ylabel="T_x (µs)", title="Phase-flip lifetime T_x")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes3[2]
    ax.semilogy(st, _smth(ratio_sc, W), label="CMA-ES", alpha=0.85)
    ax.semilogy(st, _smth(ratio_sa, W), label="Adam", linestyle="--", alpha=0.85)
    ax.axhline(
        args.nu, color="r", linestyle=":", lw=1.2, label=f"target ν={args.nu:.0f}"
    )
    ax.set(xlabel="Drift step", ylabel="T_z / T_x", title="Noise bias T_z/T_x")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3.savefig("cat_static_drift.png", dpi=150)
    print("Saved cat_static_drift.png")

    # ── Fig 4: Phase C — online stabilization ─────────────────────────────────
    ratio_oc = Tz_oc / np.clip(Tx_oc, 1e-9, None)
    ratio_oa = Tz_oa / np.clip(Tx_oa, 1e-9, None)

    fig4, axes4 = plt.subplots(2, 3, figsize=(16, 10))
    fig4.suptitle(
        "Phase C — Online stabilization  (continuous re-optimization tracks drift)",
        fontsize=12,
    )

    # Row 1: Lifetimes and bias — online vs static overlaid for direct comparison
    ratio_ppo = Tz_ppo / np.clip(Tx_ppo, 1e-9, None)

    ax = axes4[0, 0]
    ax.semilogy(st, _smth(Tz_oc, W), label="CMA-ES online")
    ax.semilogy(st, _smth(Tz_oa, W), label="Adam online", linestyle="--")
    ax.semilogy(st, _smth(Tz_ppo, W), label="PPO", linestyle="-.", color="C4")
    ax.semilogy(
        st,
        _smth(Tz_sc, W),
        label="CMA-ES static",
        alpha=0.35,
        color="C0",
        linestyle=":",
    )
    ax.semilogy(
        st, _smth(Tz_sa, W), label="Adam static", alpha=0.35, color="C1", linestyle=":"
    )
    ax.set(xlabel="Drift step", ylabel="T_z (µs)", title="T_z  (online vs static)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes4[0, 1]
    ax.semilogy(st, _smth(Tx_oc, W), label="CMA-ES online")
    ax.semilogy(st, _smth(Tx_oa, W), label="Adam online", linestyle="--")
    ax.semilogy(st, _smth(Tx_ppo, W), label="PPO", linestyle="-.", color="C4")
    ax.semilogy(
        st,
        _smth(Tx_sc, W),
        label="CMA-ES static",
        alpha=0.35,
        color="C0",
        linestyle=":",
    )
    ax.semilogy(
        st, _smth(Tx_sa, W), label="Adam static", alpha=0.35, color="C1", linestyle=":"
    )
    ax.set(xlabel="Drift step", ylabel="T_x (µs)", title="T_x  (online vs static)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes4[0, 2]
    ax.semilogy(st, _smth(ratio_oc, W), label="CMA-ES online")
    ax.semilogy(st, _smth(ratio_oa, W), label="Adam online", linestyle="--")
    ax.semilogy(st, _smth(ratio_ppo, W), label="PPO", linestyle="-.", color="C4")
    ax.semilogy(
        st,
        _smth(ratio_sc, W),
        label="CMA-ES static",
        alpha=0.35,
        color="C0",
        linestyle=":",
    )
    ax.semilogy(
        st,
        _smth(ratio_sa, W),
        label="Adam static",
        alpha=0.35,
        color="C1",
        linestyle=":",
    )
    ax.axhline(
        args.nu, color="r", linestyle=":", lw=1.2, label=f"target ν={args.nu:.0f}"
    )
    ax.set(
        xlabel="Drift step", ylabel="T_z / T_x", title="Noise bias  (online vs static)"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 2: Parameter tracking — shows how online methods compensate for drift
    # Ideal compensation: nominal_param = static_optimum − drift(t)
    ideal_g2_corr = -drift_traj[:, 0]  # ideal correction in g2
    ideal_eps_corr = -drift_traj[:, 2]  # ideal correction in eps_d

    ax = axes4[1, 0]
    ax.plot(st, params_oc[:, 0], label="CMA-ES nominal Re(g₂)")
    ax.plot(st, params_oa[:, 0], label="Adam nominal Re(g₂)", linestyle="--")
    ax.plot(
        st,
        np.full(args.n_steps, x_cmaes[0]),
        "k:",
        lw=0.8,
        label=f"Static optimum ({x_cmaes[0]:.3f})",
    )
    ax.plot(
        st,
        x_cmaes[0] + ideal_g2_corr,
        "k--",
        lw=0.8,
        alpha=0.5,
        label="Ideal (static − drift)",
    )
    ax.set(xlabel="Drift step", ylabel="Re(g₂) [MHz]", title="Re(g₂) nominal set point")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes4[1, 1]
    ax.plot(st, params_oc[:, 2], label="CMA-ES nominal Re(ε_d)")
    ax.plot(st, params_oa[:, 2], label="Adam nominal Re(ε_d)", linestyle="--")
    ax.plot(
        st,
        np.full(args.n_steps, x_cmaes[2]),
        "k:",
        lw=0.8,
        label=f"Static optimum ({x_cmaes[2]:.3f})",
    )
    ax.plot(
        st,
        x_cmaes[2] + ideal_eps_corr,
        "k--",
        lw=0.8,
        alpha=0.5,
        label="Ideal (static − drift)",
    )
    ax.set(
        xlabel="Drift step", ylabel="Re(ε_d) [MHz]", title="Re(ε_d) nominal set point"
    )
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes4[1, 2]
    # Correction applied = current_nominal − static_optimum
    # Ideal correction = −drift(t)   (perfectly cancels drift)
    corr_oc_g2 = params_oc[:, 0] - x_cmaes[0]
    corr_oa_g2 = params_oa[:, 0] - x_adam[0]
    ax.plot(st, corr_oc_g2, label="CMA-ES correction Re(g₂)")
    ax.plot(st, corr_oa_g2, label="Adam correction Re(g₂)", linestyle="--")
    ax.plot(st, ideal_g2_corr, "k:", lw=1.2, label="Ideal: −drift")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set(
        xlabel="Drift step",
        ylabel="Δ Re(g₂) [MHz]",
        title="g₂ correction vs ideal  (−drift = perfect)",
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig4.tight_layout()
    fig4.savefig("cat_online_stab.png", dpi=150)
    print("Saved cat_online_stab.png")

    # ── Fig 5: Feedback parameter trajectories ────────────────────────────────
    # "Effective" = what the system actually experiences = nominal + drift
    # Ideal effective = static optimum (constant) — perfect feedback would hold this
    # Naive (no feedback) = x_static + drift = drifting away from optimum
    x_eff_oc = np.clip(params_oc + drift_traj, BOUNDS[:, 0], BOUNDS[:, 1])
    x_eff_oa = np.clip(params_oa + drift_traj, BOUNDS[:, 0], BOUNDS[:, 1])
    x_eff_pc = np.clip(params_pc + drift_traj, BOUNDS[:, 0], BOUNDS[:, 1])
    x_eff_pa = np.clip(params_pa + drift_traj, BOUNDS[:, 0], BOUNDS[:, 1])
    x_eff_ppo = np.clip(params_ppo + drift_traj, BOUNDS[:, 0], BOUNDS[:, 1])
    # Static (no feedback): operating point drifts with hardware
    x_eff_sc = np.clip(x_cmaes[None, :] + drift_traj, BOUNDS[:, 0], BOUNDS[:, 1])

    fig5, axes5 = plt.subplots(2, 2, figsize=(14, 9))
    fig5.suptitle(
        "Effective parameter trajectories under real-time feedback\n"
        f"(effective = nominal set-point + hardware drift, "
        f"drift_period={args.drift_period}, optimizer_freq={args.optimizer_freq})",
        fontsize=11,
    )

    ls = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    labels = [
        "CMA-ES standard",
        "Adam standard",
        "CMA-ES predictive",
        "Adam predictive",
        "PPO",
    ]
    x_effs_g2 = [
        x_eff_oc[:, 0],
        x_eff_oa[:, 0],
        x_eff_pc[:, 0],
        x_eff_pa[:, 0],
        x_eff_ppo[:, 0],
    ]
    x_effs_eps = [
        x_eff_oc[:, 2],
        x_eff_oa[:, 2],
        x_eff_pc[:, 2],
        x_eff_pa[:, 2],
        x_eff_ppo[:, 2],
    ]

    ax = axes5[0, 0]
    for lbl, ls_, xe in zip(labels, ls, x_effs_g2):
        ax.plot(st, _smth(xe, W), linestyle=ls_, label=lbl)
    ax.plot(st, x_eff_sc[:, 0], "k:", lw=0.8, alpha=0.5, label="Static (no feedback)")
    ax.axhline(
        x_cmaes[0],
        color="k",
        lw=1.2,
        linestyle="--",
        label=f"Ideal target ({x_cmaes[0]:.3f})",
    )
    ax.set(
        xlabel="Drift step",
        ylabel="Re(g₂) effective [MHz]",
        title="Effective g₂ — feedback keeps it near target",
    )
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes5[0, 1]
    for lbl, ls_, xe in zip(labels, ls, x_effs_eps):
        ax.plot(st, _smth(xe, W), linestyle=ls_, label=lbl)
    ax.plot(st, x_eff_sc[:, 2], "k:", lw=0.8, alpha=0.5, label="Static (no feedback)")
    ax.axhline(
        x_cmaes[2],
        color="k",
        lw=1.2,
        linestyle="--",
        label=f"Ideal target ({x_cmaes[2]:.3f})",
    )
    ax.set(
        xlabel="Drift step",
        ylabel="Re(ε_d) effective [MHz]",
        title="Effective ε_d — feedback keeps it near target",
    )
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes5[1, 0]
    for lbl, ls_, xe in zip(labels, ls, x_effs_g2):
        err = xe - x_cmaes[0]
        ax.plot(st, _smth(err, W), linestyle=ls_, label=lbl)
    ax.plot(st, x_eff_sc[:, 0] - x_cmaes[0], "k:", lw=0.8, alpha=0.5, label="Static")
    ax.axhline(0, color="k", lw=1.0)
    ax.set(
        xlabel="Drift step",
        ylabel="Error in Re(g₂) [MHz]",
        title="g₂ tracking error  (0 = perfect compensation)",
    )
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes5[1, 1]
    for lbl, ls_, xe in zip(labels, ls, x_effs_eps):
        err = xe - x_cmaes[2]
        ax.plot(st, _smth(err, W), linestyle=ls_, label=lbl)
    ax.plot(st, x_eff_sc[:, 2] - x_cmaes[2], "k:", lw=0.8, alpha=0.5, label="Static")
    ax.axhline(0, color="k", lw=1.0)
    ax.set(
        xlabel="Drift step",
        ylabel="Error in Re(ε_d) [MHz]",
        title="ε_d tracking error  (0 = perfect compensation)",
    )
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig5.tight_layout()
    fig5.savefig("cat_feedback_params.png", dpi=150)
    print("Saved cat_feedback_params.png")

    # ── Fig 6: Lifetime trajectories under real-time feedback ─────────────────
    fig6, axes6 = plt.subplots(1, 3, figsize=(16, 5))
    fig6.suptitle(
        "System property trajectories under real-time feedback\n"
        f"(drift_period={args.drift_period} steps, "
        f"optimizer_freq={args.optimizer_freq}×, lookahead={args.lookahead})",
        fontsize=11,
    )

    Tz_sets = [Tz_oc, Tz_oa, Tz_pc, Tz_pa, Tz_ppo, Tz_sc]
    Tx_sets = [Tx_oc, Tx_oa, Tx_pc, Tx_pa, Tx_ppo, Tx_sc]
    lbls6 = [
        "CMA-ES std",
        "Adam std",
        "CMA-ES pred",
        "Adam pred",
        "PPO",
        "Static (no FB)",
    ]
    ls6 = ["-", "--", "-.", ":", "-.", ":"]
    alphas6 = [1.0, 1.0, 1.0, 1.0, 1.0, 0.45]
    colors6 = ["C0", "C1", "C2", "C3", "C4", "gray"]

    ax = axes6[0]
    for Tz, lbl, ls_, al, co in zip(Tz_sets, lbls6, ls6, alphas6, colors6):
        ax.semilogy(st, _smth(Tz, W), label=lbl, linestyle=ls_, alpha=al, color=co)
    ax.set(
        xlabel="Drift step",
        ylabel="T_z (µs)",
        title="Bit-flip lifetime T_z\n(higher = better; flat = stable feedback)",
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes6[1]
    for Tx, lbl, ls_, al, co in zip(Tx_sets, lbls6, ls6, alphas6, colors6):
        ax.semilogy(st, _smth(Tx, W), label=lbl, linestyle=ls_, alpha=al, color=co)
    ax.set(
        xlabel="Drift step",
        ylabel="T_x (µs)",
        title="Phase-flip lifetime T_x\n(flat = stable; too low = bias lost)",
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes6[2]
    for Tz, Tx, lbl, ls_, al, co in zip(Tz_sets, Tx_sets, lbls6, ls6, alphas6, colors6):
        ratio = Tz / np.clip(Tx, 1e-9, None)
        ax.semilogy(st, _smth(ratio, W), label=lbl, linestyle=ls_, alpha=al, color=co)
    ax.axhline(
        args.nu, color="r", linestyle=":", lw=1.5, label=f"target ν={args.nu:.0f}"
    )
    ax.set(
        xlabel="Drift step",
        ylabel="T_z / T_x",
        title="Noise bias T_z/T_x\n(flat near target = good bias + stability)",
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig6.tight_layout()
    fig6.savefig("cat_feedback_lifetime.png", dpi=150)
    print("Saved cat_feedback_lifetime.png")

    plt.show()


if __name__ == "__main__":
    main()
