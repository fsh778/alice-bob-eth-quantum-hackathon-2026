# Alice & Bob x ETH Quantum Hackathon 2026 Challenge - Cat Qubit Online Stabilization under Parameter Drift

## Overview

The goal is to **design and benchmark an online optimization algorithm** for cat qubit stabilization that maintains performance under hardware drift. The drift is over longer periods of time, such that the adaptation is done before every experiment (epoch).

Because cat qubits are biased qubits, your optimizer must pursue two objectives simultaneously:
- Achieve a target bias $\eta = T_Z / T_X$
- Maximize the absolute values of $T_X$ and $T_Z$

## Repository Structure

/challenge contains all the resources (by Alice and Bob)

/solution contains our simulation files

\begin{list}
  \item naive_implementation.ipynb:  cmaes optimizer (not online), no drift
  \item naive_cmaes_drift.ipynb:  cmaes optimizer (not online), drift -> optimizer confused (naive_cmaes_drift_results)
  -> this also implemented theoretically robust implementation of \sigma_z (\Hat{X} = 1/\sqrt{2}(a^{\dag}+a))and \sigma_x (parity) and 5 and 2 point linear fits as proxys for the exponential decay to fint T_z and T_x respectively.
  \item 
  
\end{list}

## Methods

We defined the reward function to be $\log{T_z} + \log{T_x} - \lambda|T_z/T_x - bias_{const}|$. log bc. makes it easier for optimizer to solve.

Then T_z and T_x are computed from the exponential fit to the $\langle\sigma_x\rangle and \langle\sigma_z\rangle$ decay. As a proxy for less computation, we do a linear fit to the beginning of the curve where the exponential function is $\approx 1-t/T_z$. The linear fit is performed with 5 datapoints in the T_z case and with 2 in the T_x case to account for 
