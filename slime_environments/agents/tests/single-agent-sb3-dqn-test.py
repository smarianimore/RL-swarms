import slime_environments
import gymnasium as gym
from gymnasium.utils.env_checker import check_env as gym_check_env
import stable_baselines3 as sb3
from stable_baselines3.common.env_checker import check_env as sb3_check_env
from stable_baselines3 import DQN
import json
import os

print(f"{gym.__version__=}")
print(f"{sb3.__version__=}")
print(f"{os.getcwd()=}")

PARAMS_FILE = "slime_environments/agents/tests/single-agent-sb3-dqn-learning-params.json"
with open(PARAMS_FILE) as f:
    params = json.load(f)

env = gym.make("Slime-v0", **params)
gym_check_env(env.unwrapped)

print("Environment compatible with Stable Baselines3")

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000, log_interval=4)
print("SB3 DQN sample training completed.")
