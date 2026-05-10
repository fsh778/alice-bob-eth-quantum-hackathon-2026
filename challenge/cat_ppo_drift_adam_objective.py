#!/usr/bin/env python3
"""
Multi-episode drift-aware PPO for cat-qubit stabilization.

The static Adam warm start still comes from `cat_lifetime_opt.make_loss_fn`.
For PPO itself, we use a stabilization objective that better matches the
control requirement:

  - strongly prefer large Tz
  - give only limited credit for increasing Tx
  - penalize violating the ratio condition Tz / Tx >= nu
  - do not penalize oversatisfying that ratio condition

Key upgrades over the earlier single-episode script:
  - longer default training horizon per drift episode
  - more PPO updates per drift step
  - more PPO epochs per rollout
  - training across many episodes with fresh drift each time
  - actor/critic weights persist across episodes
  - x_nom resets to the Adam optimum at the start of each episode
  - held-out evaluation every few episodes to test generalization
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

CHALLENGE_DIR = Path("/Users/kameronlalee/Documents/QEC_Hackathon/challenge")
sys.path.insert(0, str(CHALLENGE_DIR))

from cat_lifetime_opt import make_loss_fn

jax.config.update("jax_enable_x64", True)

DRIFT_PERIOD = 300
DRIFT_AMP_G2 = 0.25
DRIFT_AMP_EPS = 0.60
DRIFT_PHASE = np.pi / 3
OU_TAU = 20
OU_STD_G2 = 0.020
OU_STD_EPS = 0.040

BOUNDS = np.array([[1.2, 3.5], [-0.5, 0.5], [2.0, 9.5], [-0.5, 0.5]])
BOUNDS_LO = jnp.array(BOUNDS[:, 0])
BOUNDS_HI = jnp.array(BOUNDS[:, 1])
BOUNDS_CENTER = jnp.array((BOUNDS[:, 0] + BOUNDS[:, 1]) / 2.0)
BOUNDS_SCALE = jnp.array((BOUNDS[:, 1] - BOUNDS[:, 0]) / 2.0)


def generate_drift(n_steps, seed=0, period=DRIFT_PERIOD):
    rng = np.random.default_rng(seed)
    ou_g2 = ou_eps = 0.0
    dt = 1.0 / OU_TAU
    traj = np.zeros((n_steps, 4))
    for i in range(n_steps):
        slow_g2 = DRIFT_AMP_G2 * np.sin(2 * np.pi * i / period)
        slow_eps = DRIFT_AMP_EPS * np.sin(2 * np.pi * i / period + DRIFT_PHASE)
        ou_g2 = ou_g2 * (1 - dt) + OU_STD_G2 * rng.standard_normal()
        ou_eps = ou_eps * (1 - dt) + OU_STD_EPS * rng.standard_normal()
        traj[i, 0] = slow_g2 + ou_g2
        traj[i, 2] = slow_eps + ou_eps
    return traj


def sinusoidal_only(n_steps, period=DRIFT_PERIOD):
    t = np.arange(n_steps)
    traj = np.zeros((n_steps, 4))
    traj[:, 0] = DRIFT_AMP_G2 * np.sin(2 * np.pi * t / period)
    traj[:, 2] = DRIFT_AMP_EPS * np.sin(2 * np.pi * t / period + DRIFT_PHASE)
    return traj


def run_static_adam(loss_fn, x0, n_steps=120, lr=0.02):
    def loss_with_aux(x):
        loss, Tx, Tz = loss_fn(x)
        return loss, (Tx, Tz)

    grad_fn = jax.jit(jax.value_and_grad(loss_with_aux, has_aux=True))
    x = jnp.array(x0, dtype=jnp.float64)
    m = v = jnp.zeros(4)
    hist = {"loss": [], "Tx": [], "Tz": []}

    for t in range(1, n_steps + 1):
        (lv, (Tv, Zv)), g = grad_fn(x)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g**2
        m_hat = m / (1.0 - 0.9**t)
        v_hat = v / (1.0 - 0.999**t)
        x = x - lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)
        x = jnp.clip(x, BOUNDS_LO, BOUNDS_HI)
        hist["loss"].append(float(lv))
        hist["Tx"].append(float(Tv))
        hist["Tz"].append(float(Zv))

    return np.array(x), {k: np.array(v) for k, v in hist.items()}


def evaluate_static(batched_loss, x_fixed, drift_traj):
    x_all = np.clip(x_fixed[None, :] + drift_traj, BOUNDS[:, 0], BOUNDS[:, 1])
    losses, Txs, Tzs = batched_loss(jnp.array(x_all))
    return np.array(losses), np.array(Txs), np.array(Tzs)


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
        lambda p, mi, vi: p - lr * mi / (jnp.sqrt(vi) + eps), params, mh, vh
    )
    return params, m, v


def _gae(rewards, values, next_value, gamma, lam):
    adv = np.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        nv = next_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * nv - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    return adv, adv + values


def _sm(x, w=7):
    return x if w <= 1 else np.convolve(x, np.ones(w) / w, mode="same")


def control_loss_from_metrics(
    Tx,
    Tz,
    nu,
    tx_weight=0.20,
    ratio_penalty=40.0,
    max_log_tx_credit=1.5,
):
    """
    PPO control loss.

    Why this differs from the Adam objective:
      The symmetric abs(nu - Tz/Tx) term treats "ratio too high" and
      "ratio too low" as equally bad, while +log(Tx) can reward pathological
      huge-Tx / low-Tz states. Here we only penalize ratio shortfall, because
      the actual requirement is to satisfy Tz/Tx >= nu.
    """
    Tx = jnp.clip(Tx, 1e-9, 1e9)
    Tz = jnp.clip(Tz, 1e-6, 1e9)
    ratio = Tz / Tx
    ratio_shortfall = jnp.maximum(nu - ratio, 0.0) / jnp.maximum(nu, 1.0)
    tx_credit = jnp.minimum(jnp.log(Tx), max_log_tx_credit)
    return -jnp.log(Tz) - tx_weight * tx_credit + ratio_penalty * ratio_shortfall**2


def summarize_control_metrics(Tx, Tz, nu):
    ratio = Tz / np.clip(Tx, 1e-9, None)
    shortfall = np.maximum(nu - ratio, 0.0)
    return ratio, shortfall


def build_ppo_system(
    loss_fn,
    hidden_size,
    action_scale,
    horizon,
    seed,
    control_cfg,
    log_std_min,
    log_std_max,
):
    eval_loss = jax.jit(loss_fn)
    state_dim = 4 * 6 + 3
    act_dim = 4

    rng = jax.random.PRNGKey(seed)
    rng, k1, k2 = jax.random.split(rng, 3)
    actor = _init_mlp(k1, state_dim, hidden_size, act_dim * 2)
    critic = _init_mlp(k2, state_dim, hidden_size, 1)

    actor_m = jax.tree_util.tree_map(jnp.zeros_like, actor)
    actor_v = jax.tree_util.tree_map(jnp.zeros_like, actor)
    critic_m = jax.tree_util.tree_map(jnp.zeros_like, critic)
    critic_v = jax.tree_util.tree_map(jnp.zeros_like, critic)

    tz_mu, tz_sc = float(np.log(50.0)), 3.0
    tx_mu, tx_sc = float(np.log(0.5)), 2.0

    def make_state(x_nom, x_ref, drift, drift_prev, x_eff, prev_action, Tx, Tz, nu):
        drift_delta = drift - drift_prev
        residual = x_ref - x_eff
        ratio = Tz / jnp.clip(Tx, 1e-9, None)
        shortfall = jnp.maximum(nu - ratio, 0.0) / max(float(nu), 1.0)
        return jnp.concatenate(
            [
                (x_nom - BOUNDS_CENTER) / BOUNDS_SCALE,
                (x_eff - BOUNDS_CENTER) / BOUNDS_SCALE,
                drift / BOUNDS_SCALE,
                drift_delta / BOUNDS_SCALE,
                residual / BOUNDS_SCALE,
                prev_action / BOUNDS_SCALE,
                jnp.array(
                    [
                        (jnp.log(jnp.clip(Tz, 1e-6, 1e9)) - tz_mu) / tz_sc,
                        (jnp.log(jnp.clip(Tx, 1e-9, 1e9)) - tx_mu) / tx_sc,
                        shortfall,
                    ]
                ),
            ]
        )

    def ppo_loss(
        actor_p,
        critic_p,
        states,
        actions,
        old_logp,
        advs,
        rets,
        clip_eps,
        entropy_coef,
    ):
        out = jax.vmap(lambda s: _mlp(actor_p, s))(states)
        mu = action_scale * jnp.tanh(out[:, :act_dim])
        log_std = jnp.clip(out[:, act_dim:], log_std_min, log_std_max)
        std = jnp.exp(log_std)
        new_logp = -0.5 * jnp.sum(
            ((actions - mu) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi),
            axis=-1,
        )
        ratio = jnp.exp(jnp.clip(new_logp - old_logp, -5.0, 5.0))
        adv_n = (advs - advs.mean()) / (advs.std() + 1e-8)
        unclipped = ratio * adv_n
        clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv_n
        policy = -jnp.mean(jnp.minimum(unclipped, clipped))
        entropy = jnp.mean(
            jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1)
        )
        values = jax.vmap(lambda s: _mlp(critic_p, s))(states).squeeze(-1)
        value = 0.5 * jnp.mean((values - rets) ** 2)
        return policy - entropy_coef * entropy + value

    ppo_grad = jax.jit(jax.value_and_grad(ppo_loss, argnums=(0, 1)))
    z_state = jnp.zeros((horizon, state_dim))
    z_action = jnp.zeros((horizon, act_dim))
    z_vec = jnp.zeros(horizon)
    ppo_grad(actor, critic, z_state, z_action, z_vec, z_vec, z_vec, 0.2, 0.01)

    return {
        "eval_loss": eval_loss,
        "control_loss": lambda Tx, Tz, nu: control_loss_from_metrics(
            Tx,
            Tz,
            nu,
            tx_weight=control_cfg["tx_weight"],
            ratio_penalty=control_cfg["ratio_penalty"],
            max_log_tx_credit=control_cfg["max_log_tx_credit"],
        ),
        "make_state": make_state,
        "ppo_grad": ppo_grad,
        "actor": actor,
        "critic": critic,
        "actor_m": actor_m,
        "actor_v": actor_v,
        "critic_m": critic_m,
        "critic_v": critic_v,
        "rng": rng,
        "state_dim": state_dim,
        "act_dim": act_dim,
        "opt_step": 0,
        "action_scale": action_scale,
        "log_std_min": log_std_min,
        "log_std_max": log_std_max,
    }


def run_episode(
    ppo,
    x_ref,
    drift_traj,
    nu,
    optimizer_freq,
    horizon,
    n_ppo_epochs,
    lr_actor,
    lr_critic,
    gamma,
    gae_lambda,
    clip_eps,
    entropy_coef,
    train=True,
):
    states = np.zeros((horizon, ppo["state_dim"]))
    actions = np.zeros((horizon, ppo["act_dim"]))
    old_logp = np.zeros(horizon)
    rewards = np.zeros(horizon)
    values = np.zeros(horizon)

    x_nom = jnp.array(x_ref, dtype=jnp.float64)
    drift_prev = jnp.zeros(4, dtype=jnp.float64)
    prev_action = jnp.zeros(4, dtype=jnp.float64)
    _, Tx0, Tz0 = ppo["eval_loss"](x_nom)
    Tx_now = float(Tx0)
    Tz_now = float(Tz0)
    idx = 0

    hist = {
        "x_nom": [],
        "x_eff": [],
        "drift": [],
        "loss": [],
        "reward": [],
        "Tx": [],
        "Tz": [],
        "ratio": [],
    }

    for step, drift_np in enumerate(drift_traj):
        drift = jnp.array(drift_np, dtype=jnp.float64)
        x_eff = jnp.clip(x_nom + drift, BOUNDS_LO, BOUNDS_HI)

        for _ in range(optimizer_freq):
            state = ppo["make_state"](
                x_nom=x_nom,
                x_ref=jnp.array(x_ref),
                drift=drift,
                drift_prev=drift_prev,
                x_eff=x_eff,
                prev_action=prev_action,
                Tx=Tx_now,
                Tz=Tz_now,
                nu=nu,
            )

            actor_out = _mlp(ppo["actor"], state)
            mu = ppo["action_scale"] * jnp.tanh(actor_out[: ppo["act_dim"]])
            log_std = jnp.clip(
                actor_out[ppo["act_dim"] :], ppo["log_std_min"], ppo["log_std_max"]
            )

            if train:
                ppo["rng"], sample_key = jax.random.split(ppo["rng"])
                eps = jax.random.normal(sample_key, (ppo["act_dim"],))
                action = mu + jnp.exp(log_std) * eps
                logp = float(
                    -0.5 * jnp.sum(eps**2 + 2 * log_std + jnp.log(2 * jnp.pi))
                )
            else:
                action = mu
                logp = 0.0

            x_nom = jnp.clip(x_nom + action, BOUNDS_LO, BOUNDS_HI)
            x_eff = jnp.clip(x_nom + drift, BOUNDS_LO, BOUNDS_HI)
            _, Tx_eval, Tz_eval = ppo["eval_loss"](x_eff)
            control_loss = ppo["control_loss"](Tx_eval, Tz_eval, nu)
            reward = -float(control_loss)
            Tx_now = float(Tx_eval)
            Tz_now = float(Tz_eval)

            if train:
                states[idx] = np.array(state)
                actions[idx] = np.array(action)
                old_logp[idx] = logp
                rewards[idx] = reward
                values[idx] = float(_mlp(ppo["critic"], state)[0])
                idx += 1

            prev_action = action

            if train and idx == horizon:
                next_state = ppo["make_state"](
                    x_nom=x_nom,
                    x_ref=jnp.array(x_ref),
                    drift=drift,
                    drift_prev=drift_prev,
                    x_eff=x_eff,
                    prev_action=prev_action,
                    Tx=Tx_now,
                    Tz=Tz_now,
                    nu=nu,
                )
                next_value = float(_mlp(ppo["critic"], next_state)[0])
                advs, rets = _gae(rewards, values, next_value, gamma, gae_lambda)
                sj = jnp.array(states)
                aj = jnp.array(actions)
                lpj = jnp.array(old_logp)
                advj = jnp.array(advs)
                retj = jnp.array(rets)

                for _ in range(n_ppo_epochs):
                    ppo["opt_step"] += 1
                    _, (g_actor, g_critic) = ppo["ppo_grad"](
                        ppo["actor"],
                        ppo["critic"],
                        sj,
                        aj,
                        lpj,
                        advj,
                        retj,
                        clip_eps,
                        entropy_coef,
                    )
                    ppo["actor"], ppo["actor_m"], ppo["actor_v"] = _adam_pytree(
                        ppo["actor"],
                        g_actor,
                        ppo["actor_m"],
                        ppo["actor_v"],
                        ppo["opt_step"],
                        lr_actor,
                    )
                    ppo["critic"], ppo["critic_m"], ppo["critic_v"] = _adam_pytree(
                        ppo["critic"],
                        g_critic,
                        ppo["critic_m"],
                        ppo["critic_v"],
                        ppo["opt_step"],
                        lr_critic,
                    )
                idx = 0

        drift_prev = drift
        hist["x_nom"].append(np.array(x_nom))
        hist["x_eff"].append(np.array(x_eff))
        hist["drift"].append(np.array(drift))
        hist["loss"].append(float(control_loss))
        hist["reward"].append(reward)
        hist["Tx"].append(Tx_now)
        hist["Tz"].append(Tz_now)
        hist["ratio"].append(Tz_now / max(Tx_now, 1e-9))

        if train and step % 100 == 0:
            print(
                f"      step {step:4d}  reward={reward:8.3f}  loss={float(control_loss):8.3f}  "
                f"Tz={Tz_now:8.2f}  Tx={Tx_now:8.4f}"
            )

    return {k: np.array(v) for k, v in hist.items()}


def summarize_episode(hist, nu):
    ratio_shortfall = np.maximum(nu - hist["ratio"], 0.0)
    return {
        "reward_mean": float(np.mean(hist["reward"])),
        "reward_min": float(np.min(hist["reward"])),
        "Tz_mean": float(np.mean(hist["Tz"])),
        "Tz_min": float(np.min(hist["Tz"])),
        "Tx_mean": float(np.mean(hist["Tx"])),
        "ratio_shortfall_mean": float(np.mean(ratio_shortfall)),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--nu", type=float, default=200.0)
    parser.add_argument("--lam", type=float, default=0.2)
    parser.add_argument("--n-steps", type=int, default=1200)
    parser.add_argument("--n-episodes", type=int, default=12)
    parser.add_argument("--eval-every", type=int, default=3)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--adam-steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--drift-period", type=int, default=DRIFT_PERIOD)
    parser.add_argument("--drift-seed", type=int, default=0)
    parser.add_argument("--opt-freq", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=8)
    parser.add_argument("--lr-actor", type=float, default=5e-4)
    parser.add_argument("--lr-critic", type=float, default=1.5e-3)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--entropy-coef", type=float, default=0.003)
    parser.add_argument("--action-scale", type=float, default=0.08)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--tx-weight", type=float, default=0.20)
    parser.add_argument("--ratio-penalty", type=float, default=40.0)
    parser.add_argument("--max-log-tx-credit", type=float, default=1.5)
    parser.add_argument("--log-std-min", type=float, default=-5.0)
    parser.add_argument("--log-std-max", type=float, default=-3.0)
    parser.add_argument("--smooth", type=int, default=11)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    if gpus:
        jax.config.update("jax_default_device", gpus[-1])
        print(f"Running on GPU: {gpus[-1]}")
    else:
        print("No GPU detected, running on CPU")

    loss_fn = make_loss_fn(args.nu, args.lam)
    batched_loss = jax.jit(jax.vmap(loss_fn))

    print("JIT warm-up …")
    batched_loss(jnp.ones((2, 4)))[0].block_until_ready()
    print("Done.\n")

    x0 = np.array([2.0, 0.0, 8.0, 0.0])

    print(f"Phase A: static Adam ({args.adam_steps} steps)")
    x_static, _ = run_static_adam(loss_fn, x0, n_steps=args.adam_steps, lr=args.lr)
    loss_static0, Tx_static0, Tz_static0 = loss_fn(jnp.array(x_static))
    print(
        f"  x_static={x_static}  loss={float(loss_static0):.3f}  "
        f"Tz={float(Tz_static0):.2f}  Tx={float(Tx_static0):.4f}"
    )

    print("\nBaseline on first training drift seed")
    drift0 = generate_drift(args.n_steps, seed=args.drift_seed, period=args.drift_period)
    slow0 = sinusoidal_only(args.n_steps, period=args.drift_period)
    loss_static, Tx_static, Tz_static = evaluate_static(batched_loss, x_static, drift0)
    control_loss_static = np.array(
        control_loss_from_metrics(
            jnp.array(Tx_static),
            jnp.array(Tz_static),
            args.nu,
            tx_weight=args.tx_weight,
            ratio_penalty=args.ratio_penalty,
            max_log_tx_credit=args.max_log_tx_credit,
        )
    )
    reward_static = -control_loss_static
    static_ratio, static_shortfall = summarize_control_metrics(
        Tx_static, Tz_static, args.nu
    )
    print(
        f"  reward mean={reward_static.mean():.3f}  "
        f"Tz mean={Tz_static.mean():.2f}  Tx mean={Tx_static.mean():.4f}  "
        f"ratio shortfall mean={np.mean(static_shortfall):.3f}"
    )

    ppo = build_ppo_system(
        loss_fn=loss_fn,
        hidden_size=args.hidden_size,
        action_scale=args.action_scale,
        horizon=args.horizon,
        seed=args.seed,
        control_cfg={
            "tx_weight": args.tx_weight,
            "ratio_penalty": args.ratio_penalty,
            "max_log_tx_credit": args.max_log_tx_credit,
        },
        log_std_min=args.log_std_min,
        log_std_max=args.log_std_max,
    )

    train_metrics = {
        "episode": [],
        "reward_mean": [],
        "Tz_mean": [],
        "Tx_mean": [],
        "ratio_shortfall_mean": [],
    }
    eval_metrics = {
        "episode": [],
        "reward_mean": [],
        "Tz_mean": [],
        "Tx_mean": [],
        "ratio_shortfall_mean": [],
    }
    final_train_hist = None
    final_eval_hist = None
    last_eval_drift = None

    print(
        f"\nPhase B: multi-episode PPO training "
        f"(episodes={args.n_episodes}, n_steps={args.n_steps}, "
        f"opt_freq={args.opt_freq}, horizon={args.horizon}, ppo_epochs={args.ppo_epochs})"
    )

    for ep in range(args.n_episodes):
        train_seed = args.drift_seed + ep
        print(f"\n  Episode {ep + 1}/{args.n_episodes}  train_seed={train_seed}")
        drift_train = generate_drift(
            args.n_steps, seed=train_seed, period=args.drift_period
        )
        train_hist = run_episode(
            ppo=ppo,
            x_ref=jnp.array(x_static),
            drift_traj=drift_train,
            nu=args.nu,
            optimizer_freq=args.opt_freq,
            horizon=args.horizon,
            n_ppo_epochs=args.ppo_epochs,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_eps=args.clip_eps,
            entropy_coef=args.entropy_coef,
            train=True,
        )
        final_train_hist = train_hist
        train_summary = summarize_episode(train_hist, args.nu)
        train_metrics["episode"].append(ep + 1)
        train_metrics["reward_mean"].append(train_summary["reward_mean"])
        train_metrics["Tz_mean"].append(train_summary["Tz_mean"])
        train_metrics["Tx_mean"].append(train_summary["Tx_mean"])
        train_metrics["ratio_shortfall_mean"].append(
            train_summary["ratio_shortfall_mean"]
        )
        print(
            f"    train  reward={train_summary['reward_mean']:.3f}  "
            f"Tz={train_summary['Tz_mean']:.2f}  Tx={train_summary['Tx_mean']:.4f}  "
            f"ratio shortfall={train_summary['ratio_shortfall_mean']:.3f}"
        )

        if (ep + 1) % args.eval_every == 0 or ep == args.n_episodes - 1:
            eval_seed = args.eval_seed + ep
            print(f"    held-out eval seed={eval_seed}")
            drift_eval = generate_drift(
                args.n_steps, seed=eval_seed, period=args.drift_period
            )
            eval_hist = run_episode(
                ppo=ppo,
                x_ref=jnp.array(x_static),
                drift_traj=drift_eval,
                nu=args.nu,
                optimizer_freq=args.opt_freq,
                horizon=args.horizon,
                n_ppo_epochs=args.ppo_epochs,
                lr_actor=args.lr_actor,
                lr_critic=args.lr_critic,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_eps=args.clip_eps,
                entropy_coef=args.entropy_coef,
                train=False,
            )
            final_eval_hist = eval_hist
            last_eval_drift = drift_eval
            eval_summary = summarize_episode(eval_hist, args.nu)
            eval_metrics["episode"].append(ep + 1)
            eval_metrics["reward_mean"].append(eval_summary["reward_mean"])
            eval_metrics["Tz_mean"].append(eval_summary["Tz_mean"])
            eval_metrics["Tx_mean"].append(eval_summary["Tx_mean"])
            eval_metrics["ratio_shortfall_mean"].append(
                eval_summary["ratio_shortfall_mean"]
            )
            print(
                f"    eval   reward={eval_summary['reward_mean']:.3f}  "
                f"Tz={eval_summary['Tz_mean']:.2f}  Tx={eval_summary['Tx_mean']:.4f}  "
                f"ratio shortfall={eval_summary['ratio_shortfall_mean']:.3f}"
            )

    print(
        f"\n{'Method':<26} {'Reward mean':>12} {'Tz mean':>10} {'Tx mean':>10} {'Shortfall':>12}"
    )
    print("─" * 78)
    print(
        f"{'Static (frozen baseline)':<26} {reward_static.mean():12.3f} "
        f"{Tz_static.mean():10.2f} {Tx_static.mean():10.4f} "
        f"{np.mean(static_shortfall):12.3f}"
    )
    if train_metrics["episode"]:
        print(
            f"{'PPO train (last episode)':<26} {train_metrics['reward_mean'][-1]:12.3f} "
            f"{train_metrics['Tz_mean'][-1]:10.2f} {train_metrics['Tx_mean'][-1]:10.4f} "
            f"{train_metrics['ratio_shortfall_mean'][-1]:12.3f}"
        )
    if eval_metrics["episode"]:
        print(
            f"{'PPO held-out eval (last)':<26} {eval_metrics['reward_mean'][-1]:12.3f} "
            f"{eval_metrics['Tz_mean'][-1]:10.2f} {eval_metrics['Tx_mean'][-1]:10.4f} "
            f"{eval_metrics['ratio_shortfall_mean'][-1]:12.3f}"
        )

    if args.no_plot:
        return

    ep_train = np.array(train_metrics["episode"])
    ep_eval = np.array(eval_metrics["episode"])
    W = args.smooth
    st = np.arange(args.n_steps)

    fig1, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig1.suptitle("Episode-level learning and generalization")
    axes[0, 0].plot(ep_train, train_metrics["reward_mean"], lw=1.5, label="train")
    if len(ep_eval) > 0:
        axes[0, 0].plot(ep_eval, eval_metrics["reward_mean"], "o-", lw=1.3, label="held-out eval")
    axes[0, 0].axhline(reward_static.mean(), color="k", linestyle=":", lw=1.0, label="static baseline")
    axes[0, 0].set(
        xlabel="Episode",
        ylabel="Mean reward",
        title="Reward = -stabilization loss",
    )
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(ep_train, train_metrics["Tz_mean"], lw=1.5, label="train")
    if len(ep_eval) > 0:
        axes[0, 1].plot(ep_eval, eval_metrics["Tz_mean"], "o-", lw=1.3, label="held-out eval")
    axes[0, 1].axhline(Tz_static.mean(), color="k", linestyle=":", lw=1.0, label="static baseline")
    axes[0, 1].set(xlabel="Episode", ylabel="Mean Tz [µs]", title="Tz over episodes")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(ep_train, train_metrics["Tx_mean"], lw=1.5, label="train")
    if len(ep_eval) > 0:
        axes[1, 0].plot(ep_eval, eval_metrics["Tx_mean"], "o-", lw=1.3, label="held-out eval")
    axes[1, 0].axhline(Tx_static.mean(), color="k", linestyle=":", lw=1.0, label="static baseline")
    axes[1, 0].set(xlabel="Episode", ylabel="Mean Tx [µs]", title="Tx over episodes")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(
        ep_train, train_metrics["ratio_shortfall_mean"], lw=1.5, label="train"
    )
    if len(ep_eval) > 0:
        axes[1, 1].plot(
            ep_eval,
            eval_metrics["ratio_shortfall_mean"],
            "o-",
            lw=1.3,
            label="held-out eval",
        )
    axes[1, 1].axhline(
        np.mean(static_shortfall),
        color="k",
        linestyle=":",
        lw=1.0,
        label="static baseline",
    )
    axes[1, 1].set(
        xlabel="Episode",
        ylabel="Mean max(nu - Tz/Tx, 0)",
        title="Ratio condition violations",
    )
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=8)
    fig1.tight_layout()
    fig1.savefig("cat_ppo_episode_learning.png", dpi=150)

    fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig2.suptitle("First-seed drift baseline")
    axes[0].plot(st, drift0[:, 0], lw=0.9, label="total")
    axes[0].plot(st, slow0[:, 0], "r--", lw=1.1, label="sinusoidal")
    axes[0].set(xlabel="Step", ylabel="Δ Re(g2) [MHz]", title="g2 drift")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].plot(st, drift0[:, 2], lw=0.9, label="total")
    axes[1].plot(st, slow0[:, 2], "r--", lw=1.1, label="sinusoidal")
    axes[1].set(xlabel="Step", ylabel="Δ Re(eps_d) [MHz]", title="eps_d drift")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig("cat_ppo_drift_observed.png", dpi=150)

    if final_eval_hist is not None and last_eval_drift is not None:
        x_eff_static_eval = np.clip(
            x_static[None, :] + last_eval_drift, BOUNDS[:, 0], BOUNDS[:, 1]
        )
        loss_static_eval, Tx_static_eval, Tz_static_eval = evaluate_static(
            batched_loss, x_static, last_eval_drift
        )
        reward_static_eval = -np.array(
            control_loss_from_metrics(
                jnp.array(Tx_static_eval),
                jnp.array(Tz_static_eval),
                args.nu,
                tx_weight=args.tx_weight,
                ratio_penalty=args.ratio_penalty,
                max_log_tx_credit=args.max_log_tx_credit,
            )
        )

        fig3, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        fig3.suptitle("Held-out evaluation on unseen drift")
        axes[0].plot(st, _sm(Tz_static_eval, W), "k:", lw=1.0, label="static")
        axes[0].plot(st, _sm(final_eval_hist["Tz"], W), lw=1.5, label="PPO")
        axes[0].set(xlabel="Step", ylabel="Tz [µs]", title="Tz")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=8)
        axes[1].plot(st, _sm(Tx_static_eval, W), "k:", lw=1.0, label="static")
        axes[1].plot(st, _sm(final_eval_hist["Tx"], W), lw=1.5, label="PPO")
        axes[1].set(xlabel="Step", ylabel="Tx [µs]", title="Tx")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=8)
        axes[2].plot(st, _sm(reward_static_eval, W), "k:", lw=1.0, label="static")
        axes[2].plot(st, _sm(final_eval_hist["reward"], W), lw=1.5, label="PPO")
        axes[2].set(
            xlabel="Step",
            ylabel="Reward",
            title="Reward = -stabilization loss",
        )
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(fontsize=8)
        fig3.tight_layout()
        fig3.savefig("cat_ppo_heldout_eval.png", dpi=150)

        fig4, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig4.suptitle("Held-out drift compensation behavior")
        axes[0, 0].plot(st, _sm(x_eff_static_eval[:, 0], W), "k:", lw=1.0, label="static effective")
        axes[0, 0].plot(st, _sm(final_eval_hist["x_eff"][:, 0], W), lw=1.5, label="PPO effective")
        axes[0, 0].axhline(x_static[0], color="r", linestyle="--", lw=1.0, label="Adam optimum")
        axes[0, 0].set(xlabel="Step", ylabel="Re(g2) [MHz]", title="Effective g2")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(fontsize=8)
        axes[0, 1].plot(st, _sm(x_eff_static_eval[:, 2], W), "k:", lw=1.0, label="static effective")
        axes[0, 1].plot(st, _sm(final_eval_hist["x_eff"][:, 2], W), lw=1.5, label="PPO effective")
        axes[0, 1].axhline(x_static[2], color="r", linestyle="--", lw=1.0, label="Adam optimum")
        axes[0, 1].set(xlabel="Step", ylabel="Re(eps_d) [MHz]", title="Effective eps_d")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(fontsize=8)
        axes[1, 0].plot(st, _sm(final_eval_hist["x_nom"][:, 0] - x_static[0], W), lw=1.5, label="PPO correction")
        axes[1, 0].plot(st, _sm(-last_eval_drift[:, 0], W), "k:", lw=1.0, label="ideal compensation")
        axes[1, 0].set(xlabel="Step", ylabel="Nominal correction [MHz]", title="g2 correction")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(fontsize=8)
        axes[1, 1].plot(st, _sm(final_eval_hist["x_nom"][:, 2] - x_static[2], W), lw=1.5, label="PPO correction")
        axes[1, 1].plot(st, _sm(-last_eval_drift[:, 2], W), "k:", lw=1.0, label="ideal compensation")
        axes[1, 1].set(xlabel="Step", ylabel="Nominal correction [MHz]", title="eps_d correction")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend(fontsize=8)
        fig4.tight_layout()
        fig4.savefig("cat_ppo_compensation.png", dpi=150)

    plt.show()


if __name__ == "__main__":
    main()
