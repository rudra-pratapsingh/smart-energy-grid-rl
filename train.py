from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env.microgrid_env import MicrogridEnv
import matplotlib.pyplot as plt
import pandas as pd

def train():
    results = []
    all_soc = []

    for i in range(3):
        print(f"\nModel {i+1} ----------------")

        seed = 42 + i
        env = make_vec_env(lambda: MicrogridEnv(), n_envs=1, seed=seed)

        model = PPO("MlpPolicy", env, verbose=1, seed=seed)
        model.learn(total_timesteps=300000)

        model.save(f"ppo_microgrid_{i+1}")

        eval_env = MicrogridEnv()
        obs, _ = eval_env.reset()

        soc_list = []
        done = False

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = eval_env.step(action)
            soc_list.append(eval_env.soc)

        cost = eval_env.total_cost
        peak = eval_env.total_peak

        print(f"Cost={cost:.2f}, Peak={peak:.2f}")

        results.append({"model": i+1, "cost": cost, "peak": peak})
        all_soc.append(soc_list)

        plt.figure()
        plt.plot(soc_list)
        plt.title(f"SOC - Model {i+1}")
        plt.savefig(f"soc_model_{i+1}.png")
        plt.close()

    df = pd.DataFrame(results)
    df.to_csv("data/results.csv", index=False)

    plt.figure()
    for i in range(3):
        plt.plot(all_soc[i], label=f"Model {i+1}")

    plt.legend()
    plt.title("SOC Comparison")
    plt.savefig("comparison.png")

if __name__ == "__main__":
    train()