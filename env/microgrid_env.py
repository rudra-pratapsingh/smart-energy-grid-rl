import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class MicrogridEnv(gym.Env):
    def __init__(self,
                 battery_capacity=10,
                 max_charge_rate=2,
                 charge_efficiency=0.95,
                 alpha=1.0,      
                 beta=0.5,       
                 gamma=2.0):     

        super(MicrogridEnv, self).__init__()

        self.battery_capacity = battery_capacity
        self.max_charge_rate = max_charge_rate
        self.charge_efficiency = charge_efficiency

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.max_steps = 48
        self.current_step = 0
        self.total_cost = 0
        self.total_peak = 0


        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.reset()

    def _generate_profiles(self):

        load_df = pd.read_csv("data/load.csv")
        solar_df = pd.read_csv("data/solar.csv")

        self.demand_profile = load_df["load"].values[:48]
        self.solar_profile = solar_df["solar"].values[:48]
        self.peak_threshold = np.percentile(self.demand_profile, 75)

        hours = range(48)
        self.price_profile = [5 if (h % 24) < 18 else 10 for h in hours] 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._generate_profiles()

        self.soc = 0.5 * self.battery_capacity
        self.current_step = 0
        self.total_cost = 0
        self.total_peak = 0

        return self._get_state(), {}

    def _get_state(self):
        return np.array([
            self.soc / self.battery_capacity,
            self.demand_profile[self.current_step] / 100,
            self.solar_profile[self.current_step] / 50,
            self.price_profile[self.current_step] / 10,
            self.current_step / 24
        ], dtype=np.float32)

    def step(self, action):
        action = action[0]

        charge_amount = action * self.max_charge_rate

        old_soc = self.soc
        self.soc += charge_amount * self.charge_efficiency

        constraint_violation = 0
        if self.soc < 0:
            constraint_violation = abs(self.soc)
            self.soc = 0
        if self.soc > self.battery_capacity:
            constraint_violation = abs(self.soc - self.battery_capacity)
            self.soc = self.battery_capacity

        demand = self.demand_profile[self.current_step]
        solar = self.solar_profile[self.current_step]
        price = self.price_profile[self.current_step]

        net_demand = demand - solar
        battery_discharge = max(-charge_amount, 0)
        battery_charge = max(charge_amount, 0)
        grid_import = net_demand - battery_discharge + battery_charge
        grid_import = max(grid_import, 0)

        cost = grid_import * price

        peak_penalty = max(grid_import - self.peak_threshold, 0)
        
        self.total_cost += cost
        self.total_peak += peak_penalty

        reward = (
            - self.alpha * cost
            - self.beta * peak_penalty
            - self.gamma * constraint_violation
        )

        if self.soc < 0.2 * self.battery_capacity:
            reward -= 5.0

        self.current_step += 1
        done = self.current_step >= self.max_steps

        if not done:
            state = self._get_state()
        else:
            state = np.zeros(5, dtype=np.float32)

        return state, reward, done, False, {}
