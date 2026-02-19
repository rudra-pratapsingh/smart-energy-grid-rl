import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env.microgrid_env import MicrogridEnv


def train_and_evaluate(beta_value):

    env = MicrogridEnv(beta=beta_value)

    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=50000)  

    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)

    return env.total_cost, env.total_peak


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

    costs = []
    peaks = []

    print("Running RL models...")

    for beta in betas:
        cost, peak = train_and_evaluate(beta)
        print(f"Beta={beta} → Cost={cost:.2f}, Peak={peak:.2f}")
        costs.append(cost)
        peaks.append(peak)

    print("\nRunning Baseline...")
    baseline_cost, baseline_peak = baseline_controller()

    print(f"Baseline → Cost={baseline_cost:.2f}, Peak={baseline_peak:.2f}")

    # Plot trade-off
    plt.figure()
    plt.scatter(peaks, costs)
    plt.scatter(baseline_peak, baseline_cost, marker="x", s=150)

    plt.xlabel("Total Peak Violations")
    plt.ylabel("Total Energy Cost")
    plt.title("Trade-Off Curve: Cost vs Peak Violations")

    plt.show()
