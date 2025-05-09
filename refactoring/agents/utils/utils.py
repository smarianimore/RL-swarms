import os
import errno
import json
import math
import datetime
import matplotlib.pyplot as plt
from typing import Optional
from tqdm import tqdm
import numpy as np
import subprocess
import cv2
from typing import Optional

def read_params(params_path: str, learning_params_path: str, visualizer_params_path: str, logger_params_path):
    params, l_params, v_params, log_params = dict(), dict(), dict(), dict()

    try:
        with open(learning_params_path) as f:
            l_params = json.load(f)
    except Exception as e:
        print(f"[ERROR] could not open learning params file: {e}")
    
    try:
        with open(params_path) as f:
            params = json.load(f)
    except Exception as e:
        print(f"[ERROR] could not open params file: {e}")

    try:
        with open(visualizer_params_path) as f:
            v_params = json.load(f)
    except Exception as e:
        print(f"[ERROR] could not open visualizer params file: {e}")
    
    try:
        with open(logger_params_path) as f:
            log_params = json.load(f)
    except Exception as e:
        print(f"[ERROR] could not open logger params file: {e}")
        
    return params, l_params, v_params, log_params

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

def setup_train(curdir: str, params: dict, l_params: dict):
    base_dir = os.path.join(curdir, "runs/train")
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)

    #run_dir = os.path.join(base_dir, "run" + '_' + datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S"))
    #if not os.path.isdir(run_dir):
    #    os.makedirs(run_dir)
    
    output_dir = os.path.join(base_dir, "train" + "_" + datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S"))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    filename = l_params['TRAIN_OUTPUT_FILE'].replace("-", "_") + '_' +\
         datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S") + ".csv"
    output_file = os.path.join(output_dir, filename)

    weights_filename = l_params["TRAIN_WEIGHTS_FILE"] + '_' +\
        params["reward_type"] + '_' + datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S") + ".npy" 
    weights_file = os.path.join(output_dir, weights_filename)

    params_filename = l_params["TRAIN_PARAMS_FILE"] + '_' +\
        datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S") + ".txt"
    params_file = os.path.join(output_dir, params_filename)

    # Q-Learning
    alpha = l_params["alpha"]  # DOC learning rate (0 learn nothing 1 learn suddenly)
    gamma = l_params["gamma"]  # DOC discount factor (0 care only bout immediate rewards, 1 care only about future ones)
    epsilon = l_params["epsilon"]  # DOC chance of random action
    epsilon_min = l_params["epsilon_min"]  # DOC chance of random action
    decay_type = l_params["decay_type"]  # DOC di quanto diminuisce epsilon ogni episode (e.g. 1500 episodes => decay = 0.9995)
    decay = l_params["decay"]  # DOC di quanto diminuisce epsilon ogni episode (e.g. 1500 episodes => decay = 0.9995)
    train_episodes = l_params["train_episodes"]
    train_log_every = l_params["TRAIN_LOG_EVERY"]

    with open(params_file, 'w') as f:
        f.write(f"{json.dumps(params, indent=2)}\n")
        f.write("----------\n")
        f.write(f"TRAIN_EPISODES = {train_episodes}\n")
        f.write(f"TRAIN_LOG_EVERY = {train_log_every}\n")
        f.write("----------\n")
        f.write(f"alpha = {alpha}\n")
        f.write(f"gamma = {gamma}\n")
        f.write(f"epsilon = {epsilon}\n")
        f.write(f"epsilon_min = {epsilon_min}\n")
        f.write(f"decay_type = {decay_type}\n")
        f.write(f"decay = {decay}\n")
        f.write("----------\n")
    
    with open(output_file, 'w') as f:
        # From NetlogoDataAnalysis: Episode, Tick, Avg cluster size X tick, Avg reward X episode, move-toward-chemical, random-walk, drop-chemical, (learner 0)-move-toward-chemical
        f.write(
            "Episode, Tick, Epsilon, Avg cluster size X episode, "
            "Cluster extra mesures (Avg), Cluster extra mesures (Std), "
            "Cluster extra mesures (Min), Cluster extra mesures (Max), "
        )
    
        for a in params["actions"]:
            f.write(f"{a}, ")

        #for l in range(params['population'], params['population'] + params['learner_population']):
        #    for a in params["actions"]:
        #        f.write(f"(learner {l})-{a}, ")

        f.write(
            "Avg reward X episode, Reward extra measures (Avg), Reward extra measures (Std), "
            "Reward extra measures (Min), Reward extra measures (Max)\n"
        )
        #f.write("Loss, Learning rate\n")

    return (
        output_dir,
        output_file,
        weights_file,
        alpha,
        gamma,
        epsilon,
        epsilon_min,
        decay_type,
        decay,
        train_episodes,
        train_log_every
    )

def get_weight_path(my_path):
    def get_last_folder_name(path):
        folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
        if len(folders) > 0:
            last_folder_name = sorted(folders)[-1]
        else:
            raise FileNotFoundError(errno.ENOENT, "No directory in", path)
        return last_folder_name
    
    if not os.path.isdir(my_path):
        raise FileNotFoundError(errno.ENOENT, "No such directory found", my_path)
    
    my_path = os.path.join(my_path, get_last_folder_name(my_path))
    EXTENSION = ".npy"
    weight_path = None
    for root, dirs, files in os.walk(my_path):
        for file in files:
            if file.endswith(EXTENSION):
                weight_path = os.path.join(root, file)

    if weight_path == None:
        raise FileNotFoundError(errno.ENOENT, "No such weights file (.npy) found in", my_path)

    return weight_path 

def setup_eval(curdir: str, l_params, params, qtable_path):
    base_dir = os.path.join(curdir, "runs/eval")
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)

    #run_dir = os.path.join(base_dir, "run" + '_' + datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S"))
    #if not os.path.isdir(run_dir):
    #    os.makedirs(run_dir)
    
    output_dir = os.path.join(base_dir, "eval" + "_" + datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S"))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    filename = l_params['EVAL_OUTPUT_FILE'].replace("-", "_") + '_' +\
         datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S") + ".csv"
    output_file = os.path.join(output_dir, filename)
    
    if qtable_path == None:
        train_dir = os.path.join(curdir, "runs/train")
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(errno.ENOENT, "No such directory", train_dir)
        weights_file = get_weight_path(train_dir)
    else:
        weights_file = qtable_path 

    params_filename = l_params["EVAL_PARAMS_FILE"] + '_' +\
        datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S") + ".txt"
    params_file = os.path.join(output_dir, params_filename)

    test_episodes = l_params["test_episodes"]
    test_log_every = l_params["TEST_LOG_EVERY"]
    alpha = l_params["alpha"]  # DOC learning rate (0 learn nothing 1 learn suddenly)
    gamma = l_params["gamma"]  # DOC discount factor (0 care only bout immediate rewards, 1 care only about future ones)
    epsilon = l_params["epsilon"]  # DOC chance of random action
    epsilon_min = l_params["epsilon_min"]  # DOC chance of random action
    decay_type = l_params["decay_type"]  # DOC di quanto diminuisce epsilon ogni episode (e.g. 1500 episodes => decay = 0.9995)
    decay = l_params["decay"]  # DOC di quanto diminuisce epsilon ogni episode (e.g. 1500 episodes => decay = 0.9995)
    
    with open(params_file, 'w') as f:
        f.write(f"{json.dumps(params, indent=2)}\n")
        f.write("----------\n")
        f.write(f"TEST_EPISODES = {test_episodes}\n")
        f.write(f"TEST_LOG_EVERY = {test_log_every}\n")
        f.write("----------\n")
        f.write(f"alpha = {alpha}\n")
        f.write(f"gamma = {gamma}\n")
        f.write(f"epsilon = {epsilon}\n")
        f.write(f"epsilon_min = {epsilon_min}\n")
        f.write(f"decay_type = {decay_type}\n")
        f.write(f"decay = {decay}\n")
        f.write(f"weights_file = {weights_file}\n")
        f.write("----------\n")
    
    with open(output_file, 'w') as f:
        # From NetlogoDataAnalysis: Episode, Tick, Avg cluster size X tick, Avg reward X episode, move-toward-chemical, random-walk, drop-chemical, (learner 0)-move-toward-chemical
        f.write(
            "Episode, Tick, Avg cluster size X episode, "
            "Cluster extra mesures (Avg), Cluster extra mesures (Std), "
            "Cluster extra mesures (Min), Cluster extra mesures (Max), "
        )
    
        for a in params["actions"]:
            f.write(f"{a}, ")

        for l in range(params['population'], params['population'] + params['learner_population']):
            for a in params["actions"]:
                f.write(f"(learner {l})-{a}, ")
        f.write(
            "Avg reward X episode, Reward extra measures (Avg), Reward extra measures (Std), "
            "Reward extra measures (Min), Reward extra measures (Max)\n"
        )

    return output_dir, output_file, weights_file, test_episodes, test_log_every
    
def calculate_epsilon(type:str, episodes:int, ticks:int, learners:int, epsilon: float, decay:float, epsilon_end:Optional[float]):
    indexes = []
    values = []
    
    pbar = tqdm(range(episodes*ticks))
    for ep in range(1, episodes + 1):
        for tick in range(1, ticks + 1):
            for agent in range(learners):
                index = agent + tick * learners + ep * ticks * learners
                indexes.append(index)
                if ep == 1 and tick == 1:
                    pass
                else:
                    if type.lower() in "normal":
                        epsilon *= decay
                    elif type.lower() == "exponential":
                        epsilon = epsilon_end + (epsilon - epsilon_end) * math.exp(-1. * ep * decay)
                    
                values.append(epsilon)
            pbar.update(1)
                
    plt.plot(indexes, values, marker='o')
    plt.xlabel('Steps')
    plt.ylabel('epsilon value')
    plt.show()
    print(f"Final value: {epsilon}")

def positional_encoding(sequence_length, d_model):
    positions = np.arange(sequence_length)[:, np.newaxis]
    angles = np.arange(d_model)[np.newaxis, :] / np.power(10000, 2 * (np.arange(d_model) // 2) / d_model)
    encoding = positions * angles

    encoding[:, 0::2] = np.sin(encoding[:, 0::2])  # Colonne pari: seno
    encoding[:, 1::2] = np.cos(encoding[:, 1::2])  # Colonne dispari: coseno

    return encoding

def update_summary(output_file, ep, params, cluster_dict, actions_dict, action_dict, reward_dict, losses, cur_lr):
    with open(output_file, 'a') as f:
        f.write(f"{ep}, {params['episode_ticks'] * ep}, {cluster_dict[str(ep)]}, {actions_dict[str(ep)]['2']}, {actions_dict[str(ep)]['0']}, {actions_dict[str(ep)]['1']}, ")
        avg_rew = 0
        
        for l in range(params['population'], params['population'] + params['learner_population']):
            avg_rew += (reward_dict[str(ep)][str(l)] / params['episode_ticks'])
            f.write(f"{action_dict[str(ep)][str(l)]['2']}, {action_dict[str(ep)][str(l)]['0']}, {action_dict[str(ep)][str(l)]['1']}, ")
        
        avg_rew /= params['learner_population']
        f.write(f"{avg_rew}, {sum(losses)/len(losses)}, {cur_lr}\n")


def calc_final_lr(base_lr, gamma, step_size, iterations, batch_size):
    print(base_lr * gamma ** ((iterations / batch_size) // step_size) )
    

def save_env_image(image, tick, output_dir, cur_ep_dir):
    assert image is not None, "Environment error: render image is None" 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if not os.path.exists(os.path.join(output_dir, "images", cur_ep_dir)):
        os.makedirs(os.path.join(output_dir, "images", cur_ep_dir))
    cv2.imwrite(os.path.join(output_dir, "images", cur_ep_dir, f"{tick}.jpg"), image)


def video_from_images(output_dir, last_ep_dir):
    subprocess.run([
            "ffmpeg", "-y", "-framerate", "30", "-i", os.path.join(output_dir, "images", last_ep_dir, "%d.jpg"), \
            '-c:v', 'libx264', '-vf', 'fps=30', '-pix_fmt', 'yuv420p', os.path.join(output_dir, "images", last_ep_dir, "video.mp4")
            ], check=True)
    
    
def calc_evaporation(learners, lay_amount, decay):
    x = 0
    for i in range(1000):
        x = x * decay + lay_amount * learners
        print(x)
        
        
if __name__ == "__main__":
    # calc_final_lr(1e-3, .9945, 1, 51200, 128)
    # calculate_epsilon("esponential", 100, 512, 100, 0.9, 20e-9, 0.0)
    calc_evaporation(100, 1, 0.8)
