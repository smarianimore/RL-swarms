from tests.slime import pettingzoo_api, slime

PARAMS_FILE = "jaxenvs/configs/slime/multi_agent_env_params.json"
EPISODES = 5
LOG_EVERY = 1

#pettingzoo_api.test_api(PARAMS_FILE)

slime.test(PARAMS_FILE, EPISODES, LOG_EVERY)
