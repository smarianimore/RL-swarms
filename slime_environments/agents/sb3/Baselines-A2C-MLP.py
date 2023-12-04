import slime_environments
import gymnasium as gym
import json
from stable_baselines3 import A2C

PARAMS_FILE = "single-agent-a2c-mlp-env-params.json"
with open(PARAMS_FILE) as f:
    params = json.load(f)
env = gym.make("Slime-v0", **params)

model = A2C('MlpPolicy', env, verbose=2)  # 2=debug
model.learn(total_timesteps=100*params['episode_ticks'])  # total env steps

obs, _ = env.reset()
for i in range(100):
    action, _state = model.predict(obs, deterministic=True)  # QUESTION "deterministic actions"? what does it mean?
    obs, reward, _, _, _ = env.step(action)
    env.render()

env.close()
