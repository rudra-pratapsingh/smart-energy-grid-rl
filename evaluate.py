import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env.microgrid_env import MicrogridEnv

def evaluate():
    env = MicrogridEnv()
    model = PPO.load("ppo_microgrid")

    obs, _ = env.reset()

    soc_list = []
    demand_list = []
    action_list = []
    cost_total = 0

    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)

        soc_list.append(env.soc)
        demand_list.append(env.demand_profile[env.current_step-1])
        action_list.append(action[0])

    print("Final SOC:", env.soc)
    print("Total Cost:", env.total_cost)
    print("Total Peak Violations:", env.total_peak)

    plt.plot(soc_list)
    plt.title("Battery SOC over Time")
    plt.show()

if __name__ == "__main__":
    evaluate()
