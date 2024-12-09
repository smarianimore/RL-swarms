import sys
import os
from tqdm import tqdm
import datetime
import argparse

import numpy as np
import random

from agents.utils.utils import read_params
from environments.slime.slime import Slime
from agents.IQLearning import iql

def create_agent(params: dict, l_params: dict, n_obs, n_actions, train):
    population = params['population']
    learner_population = params['learner_population']
    episodes =  l_params["train_episodes"] if train else l_params["test_episodes"]
    # DOC dict che tiene conto della frequenza di scelta delle action per ogni episodio {episode: {action: _, action: _, ...}}
    # Actions:
    #   0: random-walk 
    #   1: drop-chemical 
    #   2: move-toward-chemical 
    #   3: move-away-chemical 
    #   4: walk-and-drop 
    #   5: move-and-drop
    actions_dict = {
        str(ep): {
            str(ac): 0 
            for ac in range(n_actions)
        } for ep in range(1, episodes + 1)
    }  # DOC 0 = walk, 1 = lay_pheromone, 2 = follow_pheromone
    # DOC dict che tiene conto della frequenza di scelta delle action di ogni agent per ogni episodio {episode: {agent: {action: _, action: _, ...}}}
    action_dict = {
        str(ep): {
            str(ag): {
                str(ac): 0 
                for ac in range(n_actions)
            } for ag in range(population, population + learner_population)
        } for ep in range(1, episodes + 1)
    }
    # DOC dict che tiene conto della reward di ogni agente per ogni episodio {episode: {agent: _}}
    reward_dict = {
        str(ep): {
            str(ag): 0 
            for ag in range(population, population + learner_population)
        }
        for ep in range(1, episodes + 1)
    }

    if train:
        # Q-Learning
        # Q_table
        qtable = np.zeros([learner_population, n_obs, n_actions])
        alpha = l_params["alpha"]  # DOC learning rate (0 learn nothing 1 learn suddenly)
        gamma = l_params["gamma"]  # DOC discount factor (0 care only bout immediate rewards, 1 care only about future ones)
        epsilon = l_params["epsilon"]  # DOC chance of random action
        epsilon_min = l_params["epsilon_min"]  # DOC chance of random action
        decay_type = l_params["decay_type"]  # DOC di quanto diminuisce epsilon ogni episode (e.g. 1500 episodes => decay = 0.9995)
        decay = l_params["decay"]  # DOC di quanto diminuisce epsilon ogni episode (e.g. 1500 episodes => decay = 0.9995)
        return (
            qtable,
            alpha,
            gamma,
            epsilon,
            epsilon_min,
            decay_type,
            decay,
            episodes,
            actions_dict,
            action_dict,
            reward_dict,
        )
    else:
        return (
            episodes,
            actions_dict,
            action_dict,
            reward_dict,
        )

def create_logger(curdir, params, l_params, log_params, train, weights_path=None):
    from agents.utils.logger import Logger
    
    log_every = log_params["train_log_every"] if train else log_params["test_log_every"]
    deep_algo = log_params["deep_algorithm"] 
    buffer_size = log_params["buffer_size"]
    log =  Logger(
        curdir,
        params,
        l_params,
        log_params,
        train=train,
        deep_algo=deep_algo,
        buffer_size=buffer_size,
        weights_file=weights_path
    )
    return log, log_every

def main(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    curdir = os.path.dirname(os.path.abspath(__file__))
    
    params, l_params, v_params, log_params = read_params(
        args.params_path,
        args.learning_params_path,
        args.visualizer_params_path,
        args.logger_params_path
    )

    env = Slime(args.random_seed, **params)
    if args.render:
        from environments.slime.slime import SlimeVisualizer
        env_vis = SlimeVisualizer(env.W_pixels, env.H_pixels, **v_params)
    else:
        env_vis = None
    n_obs = env.observations_n()
    n_actions = env.actions_n()
    
    if args.train:
        (
            qtable,
            alpha,
            gamma,
            epsilon,
            epsilon_min,
            decay_type,
            decay,
            train_episodes,
            actions_dict,
            action_dict,
            reward_dict,
        ) = create_agent(params, l_params, n_obs, n_actions, args.train)
        logger, train_log_every = create_logger(curdir, params, l_params, log_params, args.train)

        train_start = datetime.datetime.now()
        qtable = iql.train(
            env,
            params,
            qtable,
            actions_dict,
            action_dict,
            reward_dict,
            train_episodes,
            train_log_every,
            alpha,
            gamma,
            decay_type,
            decay,
            epsilon,
            epsilon_min,
            logger,
            env_vis
        )
        train_end = datetime.datetime.now()
        logger.save_computation_time(train_end - train_start)
        print(f"Training time: {train_end - train_start}\n")
        print("Now saving the model...\n")
        #np.save(weights_file, qtable)
        logger.save_model(qtable)
        print("Model saved.")
    else:
        (
            test_episodes,
            actions_dict,
            action_dict,
            reward_dict,
        ) = create_agent(params, l_params, n_obs, n_actions, args.train)
        logger, test_log_every = create_logger(curdir, params, l_params, log_params, args.train, args.qtable_path)

        print("Loading Q-Table weights...")
        qtable = logger.load_model()
        #qtable = np.load(weights_file)
        print("Weights are loaded.\n")
                
        test_start = datetime.datetime.now()
        iql.eval(
            env,
            params,
            actions_dict,
            action_dict,
            reward_dict,
            test_episodes,
            qtable,
            test_log_every,
            logger,
            env_vis
        )
        test_end = datetime.datetime.now()
        logger.save_computation_time(test_end - test_start, train=False)
        print(f"\nTesting time: {test_end - test_start}")
   
def check_args(args):
    assert (
        args.params_path != ""
        and os.path.isfile(args.params_path)
        and args.params_path.endswith(".json")
    ), "[ERROR] params path is empty or is not a file or is not a json file"

    assert (
        args.learning_params_path != ""
        and os.path.isfile(args.learning_params_path)
        and args.learning_params_path.endswith(".json")
    ), "[ERROR] learning params path is empty or is not a file or is not a json file"
    
    assert (
        args.visualizer_params_path != ""
        and os.path.isfile(args.visualizer_params_path)
        and args.visualizer_params_path.endswith(".json")
    ), "[ERROR] visualizer params path is empty or is not a file or is not a json file"
    
    assert (
        args.logger_params_path != ""
        and os.path.isfile(args.logger_params_path)
        and args.logger_params_path.endswith(".json")
    ), "[ERROR] logger params path is empty or is not a file or is not a json file"

    if args.qtable_path != None:
        assert(args.qtable_path.endswith(".npy")), "[ERROR] qtable weights file must be a npy file" 

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
    
    parser.add_argument(
        "--learning_params_path",
        type=str,
        default="agents/IQLearning/config/learning-params.json",
        required=False
    )
    
    parser.add_argument(
        "--logger_params_path",
        type=str,
        default="agents/IQLearning/config/logger-params.json",
        required=False
    )

    parser.add_argument(
        "--qtable_path",
        type=str,
        required=False
    )
    
    parser.add_argument("--train", type=bool, default=False, required=False)

    parser.add_argument("--random_seed", type=int, default=42, required=False)
    
    parser.add_argument("--render", type=bool, default=False, required=False)
    
    args = parser.parse_args()
    print(f"{args}\n")
    if check_args(args):
        main(args)
