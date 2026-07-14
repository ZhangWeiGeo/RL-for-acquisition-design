# Deep Reinforcement Learning for Seismic Acquisition Design

This repository contains the supporting code for the paper:

**Deep reinforcement learning for seismic acquisition design in target-oriented imaging**
Wei Zhang and Mauricio D. Sacchi — Department of Physics, University of Alberta
Submitted to *The Leading Edge*

## Abstract

Seismic acquisition design requires selecting source and receiver locations from a large number of possible configurations. Conventional deterministic optimization methods may become computationally expensive for large search spaces, depend on the initial acquisition geometry, and explore only a limited subset of possible configurations. The difficulty becomes more pronounced when both sources and receivers are allowed to move, leading to a combinatorial optimization problem that is generally NP-hard. In this study, we investigate reinforcement learning (RL) as an alternative strategy for target-oriented seismic acquisition design and focus on source-position optimization while keeping the receiver geometry fixed. Under the assumption that the target-reflector image is approximately known, we formulate the source-position design as a Markov decision process and solve it using a deep Q-learning algorithm. Each state represents a candidate source configuration, each action moves one source position left or right, and the reward is defined by the change in an imaging objective function associated with point-spread functions (PSFs), thereby encouraging acquisition geometries that improve target illumination and image quality. To make the method computationally feasible, we use Kirchhoff modeling and migration with wave-equation-based amplitudes and traveltimes, and precompute PSFs for candidate source positions. Numerical examples show that deep Q-learning identifies source configurations that produce higher-quality target-reflector images than uniform and randomly sampled geometries.

## Problem Formulation

Seismic acquisition design is a large combinatorial optimization problem. For example, selecting 35 sources out of 454 candidate surface locations yields on the order of 10^49 possible configurations — exhaustive search is infeasible. The paper formulates this as a Markov decision process:

- **State**: a candidate source configuration (which candidate positions are active).
- **Action**: move one active source position by one grid point, left or right.
- **Reward**: `+10` if the PSF-based imaging objective `J` decreases relative to the previous step, `-10` otherwise.
- **Objective**: `J = || L^T L m_ref - m_ref ||_2^2`, where `L` is the Kirchhoff modeling operator, `L^T` is the Kirchhoff migration operator, and `m_ref` is a reference reflectivity model in the target region. `L^T L`, the Hessian normal operator, is approximated with Kirchhoff (traveltime/amplitude, high-frequency asymptotic Green's function) modeling and migration rather than full wave-equation modeling, and point-spread functions (PSFs) are precomputed for every candidate source position so that evaluating `J` for a given RL state only requires assembling the PSFs of the currently selected sources.

The deep Q-network used in the seismic examples is a 3-layer fully connected network (256 neurons per hidden layer), trained with epsilon-greedy exploration, experience replay, and a periodically synchronized target network (Algorithm 1 in the manuscript, following Mnih et al., 2015).

## Repository Structure

```text
.
├── Treasure-hunting-example/   # Self-contained RL teaching examples (grid-world MDP)
├── Kevin-700-600-example/      # Deep Q-learning source-placement, SAIG velocity model
└── ref_inv/                    # Deep Q-learning applied to sparse-spike reflectivity inversion
```

### 1. `Treasure-hunting-example/`

Compact, self-contained Python examples used to illustrate the RL concepts discussed in the manuscript, using a simple 5x5 treasure-hunting grid-world MDP as a teaching analogue for the acquisition-design MDP. These scripts only require `numpy`, `matplotlib`, and `torch`, and run standalone.

```text
Treasure-hunting-example/
├── Reinforcement_learning.py     # Grid-world MDP: value/policy iteration, MC control, Q-learning, DQN
├── RM1.py                        # Robbins-Monro example: noisy root finding
├── RM2.py                        # Robbins-Monro example: tanh root finding
├── mean_estimation.py            # Incremental mean-estimation comparison
├── Q-learning-1/                 # Generated grid-world figures (episodes, policies, Q-values)
├── RM1.png / RM2.png / mean_estimation_large_step.png
```

Run with:

```bash
python Treasure-hunting-example/RM1.py
python Treasure-hunting-example/RM2.py
python Treasure-hunting-example/mean_estimation.py
python Treasure-hunting-example/Reinforcement_learning.py
```

The grid-world environment is a 5x5 MDP: the agent starts in the upper-left cell and tries to reach the goal in the lower-right cell while avoiding traps, with reward `+10` at the goal, `-10` in a trap, and `-1` per ordinary move. `Reinforcement_learning.py` compares value iteration, policy iteration, Monte Carlo control, on-/off-policy tabular Q-learning, and deep Q-learning on this environment, generating the figures used in the manuscript to explain states, actions, episodes, rewards, policies, and Q-values before introducing seismic acquisition optimization.

### 2. `Kevin-700-600-example/`

Deep Q-learning source-position optimization applied to the SAIG velocity model discussed in the manuscript (140 fixed receivers with 50 m spacing, 70 sources optimized, 20 Hz source wavelet).

```text
Kevin-700-600-example/
├── DQN_opti.py                                    # SourcePlacementEnv + DQN agent (Algorithm 1)
├── main_para.py                                   # Training/prediction parameters and entry point
├── DQN_opti_test.py                                # Driver script: builds target reflectivity, runs training/prediction
├── output-nor-False-shot_num-70-random-0/         # Training run: policy-net-final.pth, loss/reward curves, PSF outputs
├── final_result1/                                 # Curated final figures used in the manuscript (uniform vs. random vs. RL-optimized geometry, PSFs, migrated images)
├── txt/                                           # Auxiliary run logs (kernel/operator dictionaries) from the underlying modeling library
├── optimize_source_arr_final1.npz, optimize_source_arr_final2.npz   # Optimized source-index arrays
├── policy-net-final.pth                           # Trained DQN weights
└── slope_hes_ref.png, slope_hes_vp.png, target_hes_ref.png, output1.png, output2.png
```

`final_result1/` contains the migration and PSF comparisons (uniform, random, and RL-optimized source distributions) that correspond to the manuscript's SAIG-model figures.

### 3. `ref_inv/`

Deep Q-learning applied to a 1D sparse-spike reflectivity inversion / denoising example (`ReflectivityEnv`), used as an additional RL environment for reference.

```text
ref_inv/
├── DQN_ref.py           # ReflectivityEnv + DQN agent
├── DQN_ref_test.py      # Driver script (Ricker wavelet, forward modeling, training/inversion)
├── output15/             # Loss/reward history, convolution-matrix, wavelet, and inversion figures
├── plot_graph.txt, imshow.txt, dot_test.txt   # Auxiliary run logs / dot-product test output
```

## Dependencies and Reproducibility Notes

`DQN_opti.py`, `DQN_opti_test.py`, `DQN_ref.py`, and `DQN_ref_test.py` import a private, in-house modeling/migration library (e.g. `lib_sys`, `lib_pylops`, `hess_info_compute`) that is **not included** in this repository. These scripts are provided for methodological reference (environment design, reward definition, DQN architecture, and training loop) rather than as a drop-in runnable package. The `txt/`, `log.txt`, `plot_graph.txt`, `imshow.txt`, and `dot_test.txt` files under `Kevin-700-600-example/` and `ref_inv/` are auxiliary diagnostic logs produced by that internal library during the runs that generated the figures; they may reference local file paths from the original computing environment.

The `Treasure-hunting-example/` scripts have no such dependency and run with only `numpy`, `matplotlib`, and `torch`:

```bash
pip install numpy matplotlib torch
```

## Notes

- Random initialization and epsilon-greedy exploration can lead to small variations in DQN/Q-learning outputs between runs.
- All images in `final_result1/` and `output15/` are the outputs used to produce the corresponding manuscript figures.

## Data and Materials Availability

The source code and scripts used to generate the seismic acquisition-design figures in the manuscript are available at this repository:
<https://github.com/ZhangWeiGeo/RL-for-acquisition-design>

## Citation

If this code is useful for your research, please cite the associated manuscript:

```bibtex
@article{zhang_sacchi_rl_acquisition,
  title   = {Deep reinforcement learning for seismic acquisition design in target-oriented imaging},
  author  = {Zhang, Wei and Sacchi, Mauricio D.},
  journal = {The Leading Edge},
  year    = {2026}
}
```
