⚡ RL-Based Energy Management in Smart Grids
📌 Project Overview

This project implements a Reinforcement Learning (RL) framework to optimize energy management in a simulated microgrid consisting of:

Battery storage

Solar generation

Real household load data

Time-varying electricity pricing

The objective is to:

Minimize total energy cost

Reduce peak grid load

Maintain sustainable battery usage

The system is modeled as a Markov Decision Process (MDP) and solved using Proximal Policy Optimization (PPO).

🎯 Problem Formulation
State Space

At each time step (hour), the agent observes:

[SOC, Demand, Solar, Price, Hour]


SOC → Battery state of charge

Demand → Real household consumption (UCI dataset)

Solar → Solar generation profile

Price → Time-of-use electricity pricing

Hour → Time index (0–47)

Action Space

Continuous action in range:

[-1, 1]


Negative → Discharge battery

Positive → Charge battery

Reward Function

The multi-objective reward is defined as:

r=−α⋅cost−β⋅peak−γ⋅constraint

Where:

Cost → Grid import × price

Peak → Grid import above threshold

Constraint → Battery limit violations

Additionally, a deep discharge penalty is added:

If SOC < 20% of capacity → extra penalty applied

Simulates battery degradation protection

📊 Dataset

Load data is derived from:

UCI Individual Household Electric Power Consumption Dataset

Minute-level data aggregated into hourly averages

48-hour simulation window

Solar generation is modeled using a realistic irradiance curve scaled to match load magnitude.

⚙️ Microgrid Model

The environment includes:

Battery capacity: 10 units

Max charge/discharge rate: 2 units

Charge efficiency applied

48-hour episode horizon

Peak threshold is dynamically computed as:

75th percentile of demand distribution

🤖 RL Algorithm

We use:

PPO (Proximal Policy Optimization)
Implementation via Stable-Baselines3.

Why PPO?

Handles continuous action spaces

Stable training behavior

Suitable for energy control problems

📈 Trade-Off Experiment

To analyze multi-objective behavior, multiple agents are trained with different peak penalty weights (β).

Metrics evaluated:

Total Energy Cost

Total Peak Violations

Baseline Controller

A rule-based controller is implemented:

Discharge when price is high

Charge when solar exceeds demand

Otherwise idle

This provides a control comparison against RL.

📊 Sample Trade-Off Results
Beta	Cost	Peak
0.1	135.38	4.28
0.5	99.54	3.18
1.0	123.22	2.98
2.0	87.56	2.96
Baseline	105.49	3.83

Observation:

Increasing β reduces peak violations

Proper tuning enables RL to outperform baseline

Demonstrates controllable multi-objective optimization

🏗 Project Structure
smart-grid-rl/
│
├── env/
│   └── microgrid_env.py
│
├── train.py
├── evaluate.py
├── tradeoff_experiment.py
├── create_load_csv.py
├── data/
│   ├── load.csv
│   └── solar.csv
│
├── requirements.txt
└── README.md

🚀 How to Run
1️⃣ Create Load Data (from UCI dataset)
python create_load_csv.py

2️⃣ Train PPO Agent
python train.py

3️⃣ Evaluate Policy
python evaluate.py

4️⃣ Run Trade-Off Experiment
python tradeoff_experiment.py

🧠 Key Insights

Proper scaling is critical when switching from synthetic to real datasets.

Peak must depend on grid import (controllable variable), not raw demand.

Reward shaping significantly impacts learned policy behavior.

Battery health constraints increase operational cost but improve sustainability.

🔮 Future Improvements

Multi-day stochastic simulation

Real solar irradiance dataset integration

Hyperparameter tuning

SAC comparison

Average performance over multiple random seeds

🎓 Academic Contribution

This project demonstrates:

MDP formulation for smart grid control

Multi-objective RL optimization

Reward shaping analysis

Baseline comparison

Trade-off curve evaluation

Sustainable battery constraint modeling