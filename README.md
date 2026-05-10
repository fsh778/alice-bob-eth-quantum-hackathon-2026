# Alice & Bob x ETH Quantum Hackathon 2026 — Cat Qubit Online Stabilization under Parameter Drift

## Overview

The goal is to **design and benchmark an online optimization algorithm** for cat qubit stabilization that maintains performance under hardware drift. Drift occurs on longer timescales, so adaptation happens before every experiment (epoch).

Because cat qubits are biased qubits, the optimizer must pursue two objectives simultaneously:
- Achieve a target bias $\eta = T_Z / T_X$
- Maximize $T_X$ and $T_Z$ in absolute terms

---

## Repository Structure

`/challenge` — all challenge resources provided by Alice & Bob, plus our solution scripts

`/solution` — early-stage and diagnostic scripts

- `naive_implementation.ipynb` — CMA-ES optimizer, no drift, used to establish a working baseline
- `naive_cmaes_drift.py` — CMA-ES under sinusoidal drift, used to demonstrate the failure mode of a non-online optimizer

`/challenge` — full solution pipeline

- `cat_lifetime_opt.py` — core loss function shared across all optimizers
- `cat_sweep.py` — Phase 1: grid search over parameter space; Phase 2: CMA-ES fine-tuning from the best grid point
- `cat_online_stab.py` — static optimization (CMA-ES and Adam), then online tracking under drift (CMA-ES and Adam)
- `cat_ppo_drift_adam_objective.py` — PPO (reinforcement learning) approach with sinusoidal + OU drift

---

## Methods

### Physical Parameter Choices

The system has four tunable knobs: the real and imaginary parts of the two-photon coupling $g_2$ and the buffer drive amplitude $\epsilon_d$, giving the vector $[Re(g_2),\, Im(g_2),\, Re(\epsilon_d),\, Im(\epsilon_d)]$.

The loss rates $\kappa_b = 10$ MHz (buffer decay) and $\kappa_a = 1$ MHz (single-photon storage loss) are fixed by fabrication and were not optimized.

Before running any optimizer, we worked out the physically meaningful parameter bounds from the adiabatic elimination of the buffer mode:

- The **adiabatic elimination** is valid only when $\kappa_b \gg 2|g_2|$. This sets an upper limit on $g_2$ (we used $g_2 \leq 3.5$ MHz, giving $2g_2/\kappa_b = 0.7$, which is approximate but workable).
- The **cat state only exists** when the two-photon drive overcomes single-photon loss: $\epsilon_d > \kappa_a \kappa_b / (8 g_2) = 1.25$ MHz at baseline $g_2 = 1$. We used $\epsilon_d \geq 2.0$ MHz as a conservative lower bound that ensures $\alpha^2 > 1$ (a meaningfully large cat).
- For the **buffer not to saturate** the drive must satisfy $\epsilon_d \ll \kappa_b = 10$ MHz. We used $\epsilon_d \leq 9.5$ MHz.
- The effective cat size follows from these as $\alpha^2 = \epsilon_d / g_2 - \kappa_a \kappa_b / (8 g_2^2)$, which was verified numerically: the full two-mode simulation matched the analytic formula to within ~10% across the search region (see `cat_alpha_comparison.png`).

Final bounds used in all optimizers: $g_2 \in [1.2, 3.5]$ MHz, $Im(g_2) \in [-0.5, 0.5]$, $\epsilon_d \in [2.0, 9.5]$ MHz, $Im(\epsilon_d) \in [-0.5, 0.5]$.

### Loss Function

All optimizers share the same loss function, which encodes two objectives:

$$\mathcal{L} = -\log T_Z - \log T_X + \lambda \left| \eta - \frac{T_Z}{T_X} \right|$$

Using logarithms keeps the landscape well-scaled across the many decades that $T_Z$ and $T_X$ can span. The bias penalty (weight $\lambda = 0.5$) penalizes deviation from a target ratio $\eta$. We used $\eta = 1000$ in the main optimization runs to push the optimizer into the large-$\alpha$ regime where $T_Z$ grows exponentially while $T_X$ decreases only algebraically.

### Measuring $T_Z$ and $T_X$ without Knowing $\alpha$

Extracting $T_Z$ and $T_X$ from the full exponential decay is expensive: it requires many simulation time steps and a scipy fit. We used two cheap proxies that avoid this.

**$T_Z$ proxy — 5-point linear fit.**
At early times the decay is $\langle \sigma_Z(t) \rangle \approx 1 - t/T_Z$, so the slope of the normalized signal gives $-1/T_Z$ directly. We measured the I-quadrature $\langle \hat{X} \rangle = (a^\dagger + a)/2$ at 5 equally spaced points over $[0, 10\,\mu\text{s}]$ and fit a straight line using the closed-form OLS formula (no scipy, runs inside JAX). Normalizing by the first point cancels the unknown $\alpha$.

**$T_X$ proxy — two-point parity ratio.**
The parity $(-1)^n$ decays as $P(t) \approx e^{-t/T_X}$, so measuring it at two times $t_1 = 0.3\,\mu\text{s}$ and $t_2 = 1.0\,\mu\text{s}$ gives $T_X = (t_2 - t_1) / \log(P(t_1)/P(t_2))$. This requires only two simulation evaluations and no $\alpha$ estimate. The parity operator $(-1)^n = e^{i\pi a^\dagger a}$ is built once and is exact.

Both proxies are JAX-native, JIT-compiled, and batched over the full optimizer population in a single call — the entire population is evaluated simultaneously as one GPU kernel.

### Optimization Pipeline

**Step 1 — Grid search (cat_sweep.py).**
We swept a $15 \times 15$ grid over $Re(g_2) \in [1.2, 3.5]$ and $Re(\epsilon_d) \in [2.0, 9.5]$ with the imaginary parts fixed at zero. The entire 225-point grid was evaluated in one batched JAX call. This produced a landscape map of $T_X$, $T_Z$, bias, and loss (see `cat_sweep_results.png`) and identified the best starting point for fine-tuning.

**Step 2 — CMA-ES fine-tuning (cat_sweep.py, cat_lifetime_opt.py).**
Starting from the best grid point, CMA-ES optimized all four knobs with a tighter initial step size ($\sigma = 0.2$, population 12). This converged to operating points with large $\alpha^2$ and lifetimes substantially above the challenge baseline.

**Step 3 — Adam gradient-based optimization (cat_online_stab.py).**
We differentiated the full Lindblad simulation with respect to the four knobs using JAX autodiff (`jax.value_and_grad`). This required no extra simulations beyond the forward pass and gave exact gradients. Adam (β₁ = 0.9, β₂ = 0.999) converged to a comparable operating point and served as the gradient-based baseline.

### Drift Model

**First stage — sinusoidal additive drift (naive_cmaes_drift.py).**
Following the pi-pulse example in the challenge notebook, we introduced a sinusoidal additive drift on $Re(\epsilon_d)$ with amplitude $\pm 2.0$ MHz and period 100 epochs. We ran 100 drift-free epochs first so CMA-ES could fully converge and its step size $\sigma$ could collapse. When the drift started at epoch 100, the collapsed $\sigma$ was too small to track swings of $\pm 2$ MHz — the optimizer was effectively locked near its converged mean. This is the expected failure mode of a naive (non-online) optimizer and is visible in `naive_cmaes_drift_results.png`.

**Second stage — online tracking under sinusoidal + OU drift (cat_online_stab.py).**
We kept the optimizer running continuously through the drift. The drift was additive on all four knobs simultaneously (period 300 steps, sinusoidal amplitudes 13% on $g_2$ and 8% on $\epsilon_d$), with superimposed Ornstein-Uhlenbeck noise (correlation time $\tau = 30$ steps, amplitude ~1–1.5%) representing correlated electronic noise. The same deterministic random seed was used for all optimizers so comparisons are fair.

Both online CMA-ES and online Adam tracked this drift. CMA-ES maintained a non-collapsed $\sigma$ by never fully converging, so it could keep chasing the moving optimum. Online Adam computed the gradient at the drifted operating point and used it to correct the nominal parameters — the chain rule automatically encodes the direction the nominal set point must shift to compensate.

**Third stage — multiplicative complex drift (cat_online_stab.py).**
We upgraded the drift model from additive shifts to complex multiplicative perturbations. Each step, $g_2$ and $\epsilon_d$ are multiplied by a time-varying complex number whose magnitude drifts sinusoidally and whose phase rotates continuously:

$$g_2^{\text{phys}} = g_2^{\text{nom}} \cdot \left[(1 + \Delta_{\text{sin}} \sin(2\pi t/T) + \delta_{\text{OU}}) \cdot e^{i 2\pi t/T}\right]$$

This simultaneously shifts the amplitude and rotates the phase of the coupling, mimicking a resonator frequency drift that the two-photon drive cannot see. The online optimizers adapted the imaginary parts of their nominal parameters to compensate the phase rotation. Results are shown in `cat_online_stab.png`.

### Reinforcement Learning — PPO (cat_ppo_drift_adam_objective.py)

We implemented a PPO agent from scratch in ~150 lines of JAX: a two-hidden-layer MLP actor-critic, He initialization, GAE advantage estimation, and a clipped surrogate objective. The agent observes the current normalized knobs and the measured log-lifetimes (no direct drift signal) and outputs parameter adjustments.

Three specific fixes were needed to stabilize training:
1. Action log-std bounded to $(-3, -2)$ to prevent step sizes that jump outside the feasible region.
2. The bias penalty clipped at 20 to prevent the ratio error from drowning the $\log T_Z$ signal.
3. $\Delta \log T_Z$ and $\Delta \log T_X$ added to the state so the policy can infer drift direction from the trend in measurements.

PPO was the most time-consuming to tune and did not reach the performance of CMA-ES or Adam within the hackathon time. It is included as a working implementation.

---

## Results

The online CMA-ES and online Adam both tracked the implemented drift across 300-step cycles, as visible in `cat_online_stab.png`. The lifetimes and bias ratio remained stable through the drift, whereas the frozen-parameter baseline degraded significantly when the drift was applied (see `cat_static_drift.png`). The parameter sweep confirmed that the loss landscape has a clear ridge at high $\epsilon_d$ and moderate $g_2$ where $T_Z$ grows exponentially while $T_X$ stays acceptable (see `cat_sweep_results.png`).

---

## How to Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Grid sweep + CMA-ES fine-tuning
python challenge/cat_sweep.py
# Online stabilization under drift
python challenge/cat_online_stab.py
# Naive drift failure demo
python solution/naive_cmaes_drift.py
```

## Dependencies

`dynamiqs`, `jax`, `cmaes`, `matplotlib`, Python 3.12

## References / Acknowledgements

Thank you to Laurent and David for helping with questions!
