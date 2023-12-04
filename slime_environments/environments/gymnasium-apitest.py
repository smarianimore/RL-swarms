import slime_environments
import gymnasium as gym
from gymnasium.utils.env_checker import check_env as gym_check_env
import json
import os

print(f"{gym.__version__=}")
print(f"{os.getcwd()=}")

PARAMS_FILE = "single-agent-env-params.json"
with open(PARAMS_FILE) as f:
    params = json.load(f)

env = gym.make("Slime-v0", **params)
gym_check_env(env.unwrapped)
print(f"Environment compatible with Gymnasium {gym.__version__=}")
