# Alice & Bob x ETH Quantum Hackathon 2026 Challenge - Cat Qubit Online Stabilization under Parameter Drift

## Overview

The goal is to **design and benchmark an online optimization algorithm** for cat qubit stabilization that maintains performance under hardware drift. The drift is over longer periods of time, such that the adaptation is done before every experiment (epoch).

Because cat qubits are biased qubits, your optimizer must pursue two objectives simultaneously:
- Achieve a target bias $\eta = T_Z / T_X$
- Maximize the absolute values of $T_X$ and $T_Z$

## Repository Structure

/challenge contains all the resources (by Alice and Bob)

/solution contains our simulation files

- naive_implementation.ipynb:  cmaes optimizer (not online), no drift
- naive_cmaes_drift.py:  cmaes optimizer (not online), drift -> optimizer confused (naive_cmaes_drift_results)
  - this also implemented theoretically robust implementation of \sigma_z (\Hat{X} = 1/\sqrt{2}(a^{\dag}+a))and \sigma_x (parity) and 5 and 2 point linear fits as proxys for the exponential decay to fint T_z and T_x respectively.
- cat_lifetime_opt.py:
- cat_online_stab.py:
- cat_ppo_drift.py:
- cat_sweep.py:

## Methods

We defined the reward function to be $\log{T_z} + \log{T_x} - \lambda|T_z/T_x - bias_{const}|$. log bc. makes it easier for optimizer to solve.

Then T_z and T_x are computed from the exponential fit to the $\langle\sigma_x\rangle and \langle\sigma_z\rangle$ decay. As a proxy for less computation, we do a linear fit to the beginning of the curve where the exponential function is $\approx 1-t/T_z$. The linear fit is performed with 5 datapoints in the T_z case and with 2 in the T_x case to account optimize the flatter curve fit for T_z as well as keeping the number of real experiments low. Therefore the loss functions can be considered efficient (it is for T_x and almost for T_z).

Furthermore, the reward function is made robust by not relying on the initial state to measure $\langle\sigma_x\rangle and \langle\sigma_z\rangle$ by not using the pauli matrices but using $\langle\sigma_x\rangle = (-1)^n$ as well as $\langle\sigma_z\rangle = X = 1/\sqrt{2}(a^{\dag}+a)$. 

The reward function is then simply defined as $\log({T_X}) + \log({T_Z}) + \lambda * \abs{\eta - \frac{T_Z}{T_X}}$. For reinforcement learning as well as the CMA-ES optimzer we used a reward function while the gradient-based ADAM optimizer was used with a loss function. For the sake of adapting this to minimization algorithms we adjusted the sign and used minimization approaches on this new loss functions. 

\TODO online optimizers

We use several different approaches to otimize the parameters of our system. First we had to think through the physically consistent and meaningful parameters in our Hamiltonian. 
To achieve a stable cat state in the storage resonator we needed to adjust the two-photon drive such that it overcomes the single-photon loss. From the adiabatic elimination relations this implied that we needed $\epsilon_d > 1.25$ MHz. This is dependent on the values for $g_2$ we allow in the end so we chose this as a lower bound. Our optimization results showed that making it even higher yields better results. 
Furthermore the adiabatic elimination breaks down under certain parameter combinations that we also explored. The decay rate of the buffer needs to be greater than the two photon coupling such that we can later adiabatically eliminate the buffer resonator. 

\TODO drift moodel

## Results
As seen in the plots the online lagorithms could well follow the implemented drift.

## How to Run
Make .venv with python 3.12 and run pip install -r requirements.txt

## Dependencies
dynamiqs, jax, cmaes, matplotlib, python3.12

## References / Acknowledgements
We want to thank claude for its contributions and explanations.

