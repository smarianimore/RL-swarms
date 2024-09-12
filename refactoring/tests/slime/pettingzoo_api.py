from jaxenvs.envs.slime.slime_env_multi_agent import Slime
import pettingzoo
from pettingzoo.test import api_test
import json

PARAMS_FILE = "jaxenvs/configs/multi_agent_env_params.json"

def test_api(file):
    
    with open(file) as f:
        params = json.load(f)
    if params["gui"]:
        render = "human"
    else:
        render = "server"
    
    env = Slime(render_mode=render, **params)
    
    api_test(env, num_cycles=100, verbose_progress=True)
    print(f"Environment compatible with Gymnasium {pettingzoo.__version__=}")

    env.close()
