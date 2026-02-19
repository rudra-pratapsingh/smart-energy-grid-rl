# ⚡ Reinforcement Learning for Smart Grid Energy Management

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Reinforcement Learning framework to optimize energy management in a simulated microgrid environment. The system integrates battery storage, renewable solar generation, real household load data, and time-varying electricity pricing — trained using **Proximal Policy Optimization (PPO)**.

---

## 📌 Table of Contents

- [Overview](#-overview)
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
- [Future Improvements](#-future-improvements)

---

## 🔍 Overview

Modern power grids are under increasing pressure to balance variable renewable generation, dynamic demand, and cost efficiency. This project builds a custom **microgrid simulator** and trains an RL agent to perform **multi-objective energy optimization** under realistic conditions.

**Objectives:**
- Minimize total energy cost
- Reduce peak grid load
- Maintain sustainable battery usage

---

## 🎯 Problem Statement

Smart grids must simultaneously handle:

| Challenge | Description |
|---|---|
| Variable Renewable Generation | Solar output fluctuates throughout the day |
| Battery Storage Constraints | Charge/discharge limits and capacity bounds |
| Fluctuating Demand | Real household consumption patterns |
| Time-of-Use Pricing | Electricity cost varies by hour |

The problem is formulated as a **Markov Decision Process (MDP)** and solved using PPO from [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).

---

## 🧠 MDP Formulation

### State Space

At each hourly time step, the agent observes a 5-dimensional state vector:

```
[SOC, Demand, Solar, Price, Hour]
```

| Variable | Description |
|---|---|
| `SOC` | Battery State of Charge |
| `Demand` | Household electricity consumption |
| `Solar` | Renewable generation output |
| `Price` | Current electricity price |
| `Hour` | Time index (0–47) |

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
| `constraint` | Battery capacity violations |

A **deep discharge penalty** is applied when:

```
SOC < 20% of battery capacity
```

This models real-world battery degradation and promotes sustainable operation.

---

## 📊 Dataset

Load data is sourced from the [UCI Individual Household Electric Power Consumption Dataset](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption):

- Minute-level data aggregated into **hourly averages**
- Covers a **48-hour simulation window**

Solar generation is modeled using a **realistic irradiance curve** scaled to match load magnitude.

> Run `python create_load_csv.py` to preprocess and generate the required CSV files.

---

## ⚙️ Microgrid Environment

| Parameter | Value |
|---|---|
| Battery Capacity | 10 units |
| Max Charge/Discharge Rate | 2 units/hour |
| Episode Length | 48 hours |
| Peak Threshold | 75th percentile of demand |
| Peak Metric | Grid import (agent-controllable) |

The environment is implemented as a custom [Gym](https://gymnasium.farama.org/) environment in `env/microgrid_env.py`.

---

## 🤖 RL Algorithm

**Algorithm: PPO (Proximal Policy Optimization)** via Stable-Baselines3

PPO was selected for this task because it:
- Handles **continuous control** problems natively
- Offers **stable training** with clipped surrogate objectives
- Is **well-suited** for energy management and resource allocation tasks

---

## 📈 Trade-Off Analysis

Multiple agents are trained with varying peak penalty weights (β) to analyze the **cost–reliability trade-off**.

| β | Cost | Peak |
|---|---|---|
| 0.1 | 135.38 | 4.28 |
| 0.5 | 99.54 | 3.18 |
| 1.0 | 123.22 | 2.98 |
| 2.0 | 87.56 | 2.96 |
| **Baseline** | **105.49** | **3.83** |

**Key observations:**
- Increasing β consistently reduces peak violations
- RL agents outperform the rule-based baseline across all β values
- Battery health constraints increase realism without significantly harming performance
- The framework enables **controllable multi-objective optimization**

---

## 🔬 Baseline Controller

A rule-based heuristic controller is included as a benchmark:

- **Discharge** during high-price hours
- **Charge** when solar generation exceeds demand
- **Idle** otherwise

This provides a transparent performance floor for evaluating RL policy quality.

---

## 🏗️ Project Structure

```
smart-grid-rl/
│
├── env/
│   └── microgrid_env.py        # Custom Gym environment
│
├── train.py                    # PPO training script
├── evaluate.py                 # Policy evaluation
├── tradeoff_experiment.py      # Multi-beta trade-off analysis
├── create_load_csv.py          # UCI dataset preprocessing
│
├── data/
│   ├── load.csv                # Processed household load data
│   └── solar.csv               # Modeled solar irradiance data
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### 1️⃣ Generate Load Data

Preprocesses the UCI dataset into hourly CSVs:

```bash
python create_load_csv.py
```

### 2️⃣ Train the RL Agent

Trains a PPO agent on the microgrid environment:

```bash
python train.py
```

### 3️⃣ Evaluate the Policy

Runs the trained policy and reports cost and peak metrics:

```bash
python evaluate.py
```

### 4️⃣ Run Trade-Off Experiment

Trains agents across multiple β values and generates comparison results:

```bash
python tradeoff_experiment.py
```

---

## 🧠 Key Insights

- **Reward shaping** strongly influences the behavior of the learned policy — poorly designed rewards lead to undesirable strategies
- **Peak penalty must be applied to grid import** (the agent-controllable variable), not raw demand
- **Proper feature scaling** is critical when switching from synthetic to real-world datasets
- **Battery sustainability constraints** create realistic cost–reliability trade-offs that mirror actual grid operation
- PPO demonstrates consistent convergence across different reward configurations

---

## 🔮 Future Improvements

- [ ] Multi-day stochastic simulation with weather variability
- [ ] Integration of real solar irradiance datasets (e.g., NREL NSRDB)
- [ ] SAC (Soft Actor-Critic) algorithm comparison
- [ ] Hyperparameter tuning via Optuna or Ray Tune
- [ ] Statistical averaging across multiple training seeds
- [ ] Demand forecasting integration as an auxiliary input

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption) for the household energy dataset
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for the PPO implementation
- [OpenAI Gymnasium](https://gymnasium.farama.org/) for the environment interface