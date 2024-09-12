import json
from jaxenvs.envs.slime.slime_env_multi_agent import Slime

PARAMS_FILE = "jaxenvs/configs/multi_agent_env_params.json"
EPISODES = 5
LOG_EVERY = 1

def test(file, episodes, log):

    with open(PARAMS_FILE) as f:
        params = json.load(f)
    if params["gui"]:
        render = "human"
    else:
        render = "server"

    env = Slime(render_mode=render, **params)

    for ep in range(1, EPISODES + 1):
        env.reset()
        print(
            f"-------------------------------------------\nEPISODE: {ep}\n-------------------------------------------")
        for tick in range(params['episode_ticks']):
            for agent in env.agent_iter(max_iter=params["learner_population"]):
                observation, reward, _ , _, info = env.last(agent)
                env.step(env.action_space(agent).sample())
            # env.evaporate_chemical()
            # env.move()
            # env._evaporate()
            # env._diffuse()
            # env.render()

    env.close()
