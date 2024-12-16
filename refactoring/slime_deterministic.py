import sys
import os
from tqdm import tqdm
import datetime

import argparse

import json
import numpy as np
import random

from environments.slime.slime import Slime
from agents.NoLearning import deterministic_policy 

def read_params(params_path: str, visualizer_params_path: str):
    params, v_params = dict(), dict()

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
    
    return params, v_params

def plot_metrics(avg_rew, avg_cluster):
    import matplotlib.pyplot as plt

    avg_rew = np.array(avg_rew)
    avg_cluster = np.array(avg_cluster)

    fig0 = plt.figure(figsize=(15, 5), dpi=200)
    ax0 = fig0.add_axes([0.08, 0.2, 0.5, 0.5])
    ax0.grid()
    ax0.set_title(f"Average reward per episode")
    ax0.set_ylabel(f"reward")
    ax0.set_xlabel(f"Log steps")
    ax0.scatter(np.arange(avg_rew.shape[0]), avg_rew)
    ax1 = fig0.add_axes([0.65, 0.2, 0.3, 0.5])
    x = np.arange(1)  # Posizioni delle etichette sull'asse x
    width = 0.05  # Larghezza delle barre
    mean = round(avg_rew.mean(), 2)
    ax1.bar(x - width*2, mean, width, label=f'Mean ({mean})', color="orange")
    std = round(avg_rew.std(), 2)
    ax1.bar(x - width, std, width, label=f'Std ({std})', color="green")
    max = round(avg_rew.max(), 2)
    ax1.bar(x, max, width, label=f'Max ({max})', color="red")
    min = round(avg_rew.min(), 2) 
    ax1.bar(x + width, min, width, label=f'Min ({min})', color="violet")
    ax1.set_xticks(x + 0.3)  # Impostiamo le posizioni delle etichette sull'asse x
    ax1.set_title('Additional metrics')
    ax1.legend()
    reward_file_name = "Reward_plot_" + datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S") + ".png"
    fig0.savefig(reward_file_name)
    
    fig1 = plt.figure(figsize=(15, 5), dpi=200)
    ax0 = fig1.add_axes([0.08, 0.2, 0.5, 0.5])
    ax0.grid()
    ax0.set_title(f"Average cluster size per episode")
    ax0.set_ylabel(f"# agents within cluster radius")
    ax0.set_xlabel(f"Log steps")
    ax0.scatter(np.arange(avg_cluster.shape[0]), avg_cluster)
    ax1 = fig1.add_axes([0.65, 0.2, 0.3, 0.5])
    x = np.arange(1)  # Posizioni delle etichette sull'asse x
    width = 0.05  # Larghezza delle barre
    mean = round(avg_cluster.mean(), 2)
    ax1.bar(x - width*2, mean, width, label=f'Mean ({mean})', color="orange")
    std = round(avg_cluster.std(), 2)
    ax1.bar(x - width, std, width, label=f'Std ({std})', color="green")
    max = round(avg_cluster.max(), 2)
    max_count = np.where(avg_cluster == max)[0].shape[0]
    ax1.bar(x, max, width, label=f'Max ({max}) (# {max_count})', color="red")
    min = round(avg_cluster.min(), 2) 
    min_count = np.where(avg_cluster == min)[0].shape[0]
    ax1.bar(x + width, min, width, label=f'Min ({min}) (# {min_count})', color="violet")
    ax1.set_xticks(x + 0.3)  # Impostiamo le posizioni delle etichette sull'asse x
    ax1.set_title('Additional metrics')
    ax1.legend()
    cluster_file_name = "Cluster_plot_" + datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S") + ".png"
    fig1.savefig(cluster_file_name)

def main(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    curdir = os.path.dirname(os.path.abspath(__file__))
    
    params, v_params = read_params(
        args.params_path,
        args.visualizer_params_path,
    )
    
    env = Slime(args.random_seed, **params)
    if args.render:
        from environments.slime.slime import SlimeVisualizer
        env_vis = SlimeVisualizer(env.W_pixels, env.H_pixels, **v_params)
    else:
        env_vis = None

    run_start = datetime.datetime.now()
    avg_rew, avg_cluster = deterministic_policy.run(env, params, args.episodes, env_vis)
    run_end = datetime.datetime.now()
    print(f"Run time: {run_end - run_start}\n")

    print("Plotting results...\n")
    plot_metrics(avg_rew, avg_cluster)
    print("Plotting is done!")

def check_args(args):
    assert (
        args.params_path != ""
        and os.path.isfile(args.params_path)
        and args.params_path.endswith(".json")
    ), "[ERROR] params path is empty or is not a file or is not a json file"
    
    assert (
        args.visualizer_params_path != ""
        and os.path.isfile(args.visualizer_params_path)
        and args.visualizer_params_path.endswith(".json")
    ), "[ERROR] visualizer params path is empty or is not a file or is not a json file"
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--params_path",
        type=str,
        default="environments/slime/config/env-params.json",
        required=False
    )
    
    parser.add_argument(
        "--visualizer_params_path",
        type=str,
        default="environments/slime/config/env_visualizer-params.json",
        required=False
    )
    
    parser.add_argument("--random_seed", type=int, default=42, required=False)
    
    parser.add_argument("--episodes", type=int, default=2000, required=False)
    
    parser.add_argument("--render", type=bool, default=False, required=False)
    
    args = parser.parse_args()
    if check_args(args):
        print("Current args: ", args)
        main(args)