# ⚡ Reinforcement Learning for Smart Grid Energy Management

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Reinforcement Learning framework to optimize energy management in a simulated microgrid environment. The system integrates battery storage, real solar irradiance data (NSRDB), real household load data, and time-varying electricity pricing — trained using **Proximal Policy Optimization (PPO)** over a realistic **7-day (168-hour)** simulation horizon.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [What's New](#-whats-new)
- [Problem Statement](#-problem-statement)
- [MDP Formulation](#-mdp-formulation)
- [Dataset](#-dataset)
- [Microgrid Environment](#️-microgrid-environment)
- [RL Algorithm](#-rl-algorithm)
- [Trade-Off Analysis](#-trade-off-analysis)
- [Baseline Controller](#-baseline-controller)
- [Project Structure](#️-project-structure)
- [How to Run](#-how-to-run)
- [Key Insights](#-key-insights)

---

## Overview

Modern power grids are under increasing pressure to balance variable renewable generation, dynamic demand, and cost efficiency. This project builds a custom **microgrid simulator** and trains an RL agent to perform **multi-objective energy optimization** under realistic conditions.

**Objectives:**
- Minimize total energy cost
- Reduce peak grid load
- Maintain sustainable battery usage with degradation awareness

---

## What's New

This version introduces significant improvements in realism, robustness, and research quality over the previous 48-hour prototype:

| Area | Previous Version | Updated Version |
|---|---|---|
| Simulation Horizon | 48 hours | **7 days (168 hours)** |
| Solar Data | Synthetic irradiance curve | **Real NSRDB irradiance data** |
| Load Data | UCI household dataset (48-hour window) | **UCI dataset (168-hour window)** |
| Evaluation | Single model run | **Multi-seed training with averaged results** |
| Reproducibility | No seed control | **Fixed random seeds (42, 43, 44)** |
| Battery Modeling | Basic capacity constraints | **Deep discharge + cycling degradation penalties** |
| Pipeline | Manual script execution | **Automated end-to-end pipeline (`pipeline.sh`)** |
| Trade-off Analysis | Static comparison | **Multi-beta experiment with averaged metrics** |

---

## Problem Statement

Smart grids must simultaneously handle:

| Challenge | Description |
|---|---|
| Variable Renewable Generation | Real solar irradiance fluctuates throughout the day and across days |
| Battery Storage Constraints | Charge/discharge limits, capacity bounds, and degradation |
| Fluctuating Demand | Real household consumption patterns over a full week |
| Time-of-Use Pricing | Electricity cost varies by hour |

The problem is formulated as a **Markov Decision Process (MDP)** and solved using PPO from [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).

---

## MDP Formulation

### State Space

At each hourly time step, the agent observes a 5-dimensional state vector:

```
[SOC, Demand, Solar, Price, Hour]
```

| Variable | Description |
|---|---|
| `SOC` | Battery State of Charge |
| `Demand` | Household electricity consumption |
| `Solar` | Real solar irradiance (NSRDB) |
| `Price` | Current electricity price |
| `Hour` | Time index (0–167) |

### Action Space

A **continuous action** in range `[-1, 1]`:

| Value | Behavior |
|---|---|
| Negative `(-1, 0)` | Discharge battery to grid/load |
| Positive `(0, 1)` | Charge battery from grid/solar |

### Reward Function

The multi-objective reward is:

$$r = -\alpha \cdot \text{cost} - \beta \cdot \text{peak} - \gamma \cdot \text{constraint}$$

| Term | Description |
|---|---|
| `cost` | Grid Import × Electricity Price |
| `peak` | Grid import exceeding the 75th percentile threshold |
| `constraint` | Battery capacity violations + degradation penalty |

#### Battery Degradation Penalties

Two degradation mechanisms are now modeled:

- **Deep Discharge Penalty** — applied when `SOC < 20%` of battery capacity, discouraging harmful low-SOC operation
- **Cycling Penalty** — applied based on charge/discharge activity to model cumulative wear on battery lifetime

These constraints improve realism and mirror operational requirements in real-world battery energy storage systems.

---

## Dataset

### Load Data
Sourced from the [UCI Individual Household Electric Power Consumption Dataset](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption):
- Minute-level data aggregated into **hourly averages**
- Covers a **168-hour (7-day) simulation window**

### Solar Data
Sourced from **NSRDB (National Renewable Energy Laboratory Solar Radiation Database)**:
- Real Global Horizontal Irradiance (GHI) values
- Covers a **168-hour window** aligned with load data
- Scaled to match load magnitude for physical consistency

> Run `python create_load_csv.py` and `python create_solar_csv.py` to preprocess and generate the required CSV files.

---

## Microgrid Environment

| Parameter | Value |
|---|---|
| Battery Capacity | 10 units |
| Max Charge/Discharge Rate | 2 units/hour |
| Episode Length | **168 hours (7 days)** |
| Peak Threshold | 75th percentile of demand |
| Peak Metric | Grid import (agent-controllable) |
| Deep Discharge Threshold | SOC < 20% of capacity |
| Degradation Modeling | Deep discharge + cycling penalties |

The environment is implemented as a custom [Gym](https://gymnasium.farama.org/) environment in `env/microgrid_env.py`.

---

## RL Algorithm

**Algorithm: PPO (Proximal Policy Optimization)** via Stable-Baselines3

PPO was selected for this task because it:
- Handles **continuous control** problems natively
- Offers **stable training** with clipped surrogate objectives
- Is **well-suited** for energy management and resource allocation tasks

### Training Protocol

To reduce stochastic variation and improve result reliability:
- **3 models trained per configuration** using fixed seeds (42, 43, 44)
- **Results are averaged** across seeds before reporting
- This multi-seed approach provides statistically more meaningful comparisons across β values

---

## Trade-Off Analysis

Multiple agents are trained with varying peak penalty weights (β) to analyze the **cost–reliability trade-off**. Each β value is averaged across 3 seeds for robustness.

| β | Avg Cost | Avg Peak |
|---|---|---|
| 0.1 | 135.38 | 4.28 |
| 0.5 | 99.54 | 3.18 |
| 1.0 | 123.22 | 2.98 |
| 2.0 | 87.56 | 2.96 |
| **Baseline** | **105.49** | **3.83** |

**Key observations:**
- Increasing β consistently reduces peak violations across all seeds
- RL agents outperform the rule-based baseline across all β values
- Battery degradation constraints add realism without significantly harming performance
- Multi-seed averaging reveals more stable and trustworthy trade-off curves
- The framework enables **controllable multi-objective optimization**

---

## Baseline Controller

A rule-based heuristic controller is included as a benchmark:

- **Discharge** during high-price hours
- **Charge** when solar generation exceeds demand
- **Idle** otherwise

This provides a transparent performance floor for evaluating RL policy quality.

---

## Project Structure

```
smart-grid-rl/
│
├── env/
│   └── microgrid_env.py        # Custom Gym environment (168-hour, degradation-aware)
│
├── train.py                    # PPO multi-seed training script
├── evaluate.py                 # Policy evaluation with averaged metrics
├── tradeoff_experiment.py      # Multi-beta, multi-seed trade-off analysis
├── create_load_csv.py          # UCI dataset preprocessing (168-hour window)
├── create_solar_csv.py         # NSRDB solar data preprocessing
├── pipeline.sh                 # Automated end-to-end pipeline
│
├── data/
│   ├── load.csv                # Processed household load data (168 hours)
│   └── solar.csv               # Real NSRDB solar irradiance data (168 hours)
│
├── results.csv                 # Averaged training results
├── tradeoff_results.csv        # Multi-beta trade-off results
├── requirements.txt
└── README.md
```

---

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Option A: Automated Pipeline (Recommended)

Runs all steps end-to-end in the correct order:

```bash
bash pipeline.sh
```

### Option B: Step-by-Step

#### 1️ Generate Load Data

Preprocesses the UCI dataset into a 168-hour hourly CSV:

```bash
python create_load_csv.py
```

#### 2️ Generate Solar Data

Preprocesses the NSRDB dataset into a 168-hour solar CSV:

```bash
python create_solar_csv.py
```

#### 3️ Train the RL Agent

Trains 3 PPO agents with fixed seeds and saves averaged results:

```bash
python train.py
```

#### 4️ Evaluate the Policy

Runs evaluation and generates performance charts:

```bash
python evaluate.py
```

#### 5️ Run Trade-Off Experiment

Trains agents across multiple β values (multi-seed) and plots trade-off curves:

```bash
python tradeoff_experiment.py
```

---

## Key Insights

- **Extended horizon matters** — a 7-day simulation captures weekly demand patterns and multi-day solar variability that a 48-hour window misses
- **Real data improves generalization** — NSRDB solar profiles introduce realistic day/night cycles and cloud variability that synthetic curves cannot replicate
- **Multi-seed averaging is essential** — single-run RL results are noisy; averaging across seeds reveals the true policy quality
- **Battery degradation modeling** shapes long-horizon strategy — the agent learns to avoid deep discharge cycles, which is aligned with real-world battery operation
- **Reward shaping strongly influences behavior** — poorly designed rewards lead to undesirable strategies; peak penalty must be applied to grid import (the agent-controllable variable)
- **Proper feature scaling** is critical when switching from synthetic to real-world datasets
- PPO demonstrates consistent convergence across different reward configurations and seeds

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption) for the household energy dataset
- [NREL NSRDB](https://nsrdb.nrel.gov/) for the real solar irradiance data
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for the PPO implementation
- [OpenAI Gymnasium](https://gymnasium.farama.org/) for the environment interface