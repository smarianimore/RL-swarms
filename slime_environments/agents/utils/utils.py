import os
import json
import datetime

def read_params(params_path:str, learning_params_path:str):
    params, l_params = dict(), dict()
    try:
        with open(learning_params_path) as f:
            l_params = json.load(f)
    except Exception as e:
        print(f"[ERROR] could not open learning params file: {e}")
    
    try:
        with open(params_path) as f:
            params = json.load(f)
    except Exception as e:
        print(f"[ERROR] could not open learning params file: {e}")
        
    return params, l_params


def state_to_int_map(obs: list):
    if sum(obs) == 0:  # DOC [False, False]
        mapped = sum(obs)  # 0
    elif sum(obs) == 2:  # DOC [True, True]
        mapped = 3
    elif int(obs[0]) == 1 and int(obs[1]) == 0:  # DOC [True, False] ==> si trova in un cluster ma non su una patch con feromone --> difficile succeda
        mapped = 1
    else:
        mapped = 2  # DOC [False, True]
    return mapped


def setup(params:dict, l_params:dict):
    
    curdir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isdir(os.path.join(curdir, "runs")):
        os.makedirs(os.path.join(curdir, "runs"))
    
    filename = l_params['OUTPUT_FILE'].replace("-", "_") + "_" + datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S") + ".csv"
    output_file = os.path.join(curdir, "runs", filename)

    # Q-Learning
    alpha = l_params["alpha"]  # DOC learning rate (0 learn nothing 1 learn suddenly)
    gamma = l_params["gamma"]  # DOC discount factor (0 care only bout immediate rewards, 1 care only about future ones)
    epsilon = l_params["epsilon"]  # DOC chance of random action
    decay = l_params["decay"]  # DOC di quanto diminuisce epsilon ogni episode (e.g. 1500 episodes => decay = 0.9995)
    train_episodes = l_params["train_episodes"]
    test_episodes = l_params["test_episodes"]
    train_log_every = l_params["TRAIN_LOG_EVERY"]
    test_log_every = l_params["TEST_LOG_EVERY"]

    with open(output_file, 'w') as f:
        f.write(f"{json.dumps(params, indent=2)}\n")
        f.write("----------\n")
        f.write(f"TRAIN_EPISODES = {train_episodes}\n")
        f.write(f"TEST_EPISODES = {test_episodes}\n")
        f.write("----------\n")
        f.write(f"alpha = {alpha}\n")
        f.write(f"gamma = {gamma}\n")
        f.write(f"epsilon = {epsilon}\n")
        f.write(f"decay = {decay}\n")
        f.write("----------\n")
        # From NetlogoDataAnalysis: Episode, Tick, Avg cluster size X tick, Avg reward X episode, move-toward-chemical, random-walk, drop-chemical, (learner 0)-move-toward-chemical
        f.write(f"Episode, Tick, Avg cluster size X tick, ")
        
        for a in l_params["actions"]:
            f.write(f"{a}, ")
        
        for l in range(params['population'], params['population'] + params['learner_population']):
            for a in l_params["actions"]:
                f.write(f"(learner {l})-{a}, ")
        f.write("Avg reward X episode\n")
    
    return output_file, alpha, gamma, epsilon, decay, train_episodes, train_log_every, test_episodes, test_log_every