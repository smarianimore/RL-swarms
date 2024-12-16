import sys
import os
from tqdm import tqdm
import datetime

import argparse

import json
import numpy as np
import random

from environments.slime.slime import Slime
from agents.NoLearning import heuristic_policy

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

def plot_avg_rew(avg_rew, avg_cluster):
    import matplotlib.pyplot as plt

    avg_rew = np.array(avg_rew)
    mean_r = [avg_rew.mean() for _ in range(avg_rew.shape[0])]
    avg_cluster = np.array(avg_cluster)
    mean_c = [avg_cluster.mean() for _ in range(avg_rew.shape[0])]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(15, 15))  # 1 riga, 2 colonne
    ax0.plot(avg_rew, label="Reward_x_episode")
    ax0.plot(mean_r, label=f"Avg_reward ({avg_rew.mean().round(2)})")
    ax0.set_xlabel("Episode")
    ax0.set_ylabel("Reward")
    ax0.legend()
    #ax0.savefig("Reward_x_episode_plot.png")
    ax1.plot(avg_cluster, label="Cluster_x_episode")
    ax1.plot(mean_c, label=f"Avg_cluster ({avg_cluster.mean().round(2)})")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("CLuster")
    ax1.legend()
    fig.savefig("Metrics_plot.png")
    #plt.show()

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
    avg_rew, avg_cluster = heuristic_policy.run(env, params, args.episodes, args.epsilon, env_vis)
    run_end = datetime.datetime.now()
    print(f"Run time: {run_end - run_start}\n")
    plot_avg_rew(avg_rew, avg_cluster)

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
    
    parser.add_argument("--epsilon", type=float, default=0.1, required=False)
    
    parser.add_argument("--render", type=bool, default=False, required=False)
    
    args = parser.parse_args()
    if check_args(args):
        print("Current args: ", args)
        main(args)