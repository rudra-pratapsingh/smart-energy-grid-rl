from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env.microgrid_env import MicrogridEnv

def train():
    env = make_vec_env(lambda: MicrogridEnv(), n_envs=1)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99
    )

    model.learn(total_timesteps=200000)

    model.save("ppo_microgrid")

if __name__ == "__main__":
    train()
