from SlimeEnvMultiAgent import Slime
import pettingzoo
from pettingzoo.test import api_test
import json

PARAMS_FILE = r"slime_environments\environments\multi-agent-env-params.json"

with open(PARAMS_FILE) as f:
    params = json.load(f)
if params["gui"]:
    render = "human"
else:
    render = "server"

env = Slime(render_mode=render, **params)
api_test(env, num_cycles=100, verbose_progress=True)
print(f"Environment compatible with Gymnasium {pettingzoo.__version__=}")
