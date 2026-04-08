import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env.microgrid_env import MicrogridEnv


def train_and_evaluate(beta, seed):
    env = make_vec_env(lambda: MicrogridEnv(beta=beta), n_envs=1, seed=seed)

    model = PPO("MlpPolicy", env, verbose=0, seed=seed)
    model.learn(total_timesteps=100000)

    eval_env = MicrogridEnv(beta=beta)
    obs, _ = eval_env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = eval_env.step(action)

    return eval_env.total_cost, eval_env.total_peak


def baseline_controller():
    env = MicrogridEnv()
    obs, _ = env.reset()
    done = False

    while not done:
        price = env.price_profile[env.current_step]
        solar = env.solar_profile[env.current_step]
        demand = env.demand_profile[env.current_step]

        if price > 8:
            action = [-1.0]
        elif solar > demand:
            action = [1.0]
        else:
            action = [0.0]

        obs, reward, done, _, _ = env.step(action)

    return env.total_cost, env.total_peak


if __name__ == "__main__":

    betas = [0.1, 0.5, 1.0, 2.0]

    results = []

    print("Running RL models...\n")

    for beta in betas:
        costs = []
        peaks = []

        print(f"Beta = {beta}")

        for i in range(3):
            seed = 42 + i

            cost, peak = train_and_evaluate(beta, seed)

            print(f"  Model {i+1} → Cost={cost:.2f}, Peak={peak:.2f}")

            costs.append(cost)
            peaks.append(peak)

        avg_cost = np.mean(costs)
        avg_peak = np.mean(peaks)

        print(f"  → Avg Cost={avg_cost:.2f}, Avg Peak={avg_peak:.2f}\n")

        results.append({
            "beta": beta,
            "cost": avg_cost,
            "peak": avg_peak
        })

    print("Running Baseline...\n")
    baseline_cost, baseline_peak = baseline_controller()

    print(f"Baseline → Cost={baseline_cost:.2f}, Peak={baseline_peak:.2f}")

    df = pd.DataFrame(results)
    df.to_csv("data/tradeoff_results.csv", index=False)

    plt.figure()
    plt.plot(df["peak"], df["cost"], marker="o", label="RL Models")
    plt.scatter(baseline_peak, baseline_cost, marker="x", s=150, label="Baseline")

    for i in range(len(df)):
        plt.text(df["peak"][i], df["cost"][i], f"β={df['beta'][i]}")

    plt.xlabel("Peak Violations")
    plt.ylabel("Cost")
    plt.title("Trade-off Curve (Cost vs Peak)")
    plt.legend()
    plt.savefig("tradeoff_curve.png")

    plt.figure()
    plt.plot(df["beta"], df["cost"], marker="o")
    plt.xlabel("Beta")
    plt.ylabel("Cost")
    plt.title("Beta vs Cost")
    plt.savefig("beta_vs_cost.png")

    plt.figure()
    plt.plot(df["beta"], df["peak"], marker="o")
    plt.xlabel("Beta")
    plt.ylabel("Peak")
    plt.title("Beta vs Peak")
    plt.savefig("beta_vs_peak.png")

    print("\nTrade-off graphs saved!")