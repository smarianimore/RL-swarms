import json
from jaxenvs.envs.slime.slime_env_multi_agent import Slime

def test(file, episodes, log):
    with open(file) as f:
        params = json.load(f)

    #if params["gui"]:
    #    render = "human"
    #else:
    #    render = "server"
    
    env = Slime(**params)

    for ep in range(1, episodes + 1):
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
