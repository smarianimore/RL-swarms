import os
import datetime
import argparse
import json
import numpy as np
import random
from environments.slime_dph_5.slime_dph import Slime
from agents.IQLearning import iql

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

def create_logger(curdir, params, l_params, log_params, train, weights_path=None):
    from agents.utils.logger import Logger
    
    log_every = log_params["train_log_every"] if train else log_params["test_log_every"]
    buffer_size = log_params["buffer_size"]
    log =  Logger(
        curdir,
        params,
        l_params,
        log_params,
        train=train,
        deep_algo=False,
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
        from environments.slime_dph_5.slime_dph import SlimeVisualizer
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
            cluster_dict,
            cluster_actions_dict,
            cluster_action_dict,
            cluster_reward_dict,
            scatter_actions_dict,
            scatter_action_dict,
            scatter_reward_dict,
        ) = iql.create_agent(params, l_params, n_obs, n_actions, args.train)
        logger, train_log_every = create_logger(curdir, params, l_params, log_params, args.train)

        PRINT_METRICS = 30
        train_start = datetime.datetime.now()
        qtable = iql.train(
            env,
            params,
            qtable,
            cluster_dict,
            cluster_actions_dict,
            cluster_action_dict,
            cluster_reward_dict,
            scatter_actions_dict,
            scatter_action_dict,
            scatter_reward_dict,
            train_episodes,
            train_log_every,
            alpha,
            gamma,
            decay_type,
            decay,
            epsilon,
            epsilon_min,
            PRINT_METRICS,
            logger,
            env_vis
        )
        train_end = datetime.datetime.now()
        logger.save_computation_time(train_end - train_start)
        print(f"Training time: {train_end - train_start}\n")
        print("Now saving the model...\n")
        logger.save_model(qtable)
        print("Model saved.")
    else:
        (
            test_episodes,
            cluster_dict,
            cluster_actions_dict,
            cluster_action_dict,
            cluster_reward_dict,
            scatter_actions_dict,
            scatter_action_dict,
            scatter_reward_dict,
        ) = iql.create_agent(params, l_params, n_obs, n_actions, args.train)
        logger, test_log_every = create_logger(curdir, params, l_params, log_params, args.train, args.qtable_path)

        print("Loading Q-Table weights...")
        qtable = logger.load_model()
        print("Weights are loaded.\n")
                
        test_start = datetime.datetime.now()
        iql.eval(
            env,
            params,
            cluster_dict,
            cluster_actions_dict,
            cluster_action_dict,
            cluster_reward_dict,
            scatter_actions_dict,
            scatter_action_dict,
            scatter_reward_dict,
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
        default="environments/slime_dph_5/config/env-params.json",
        required=False
    )

    parser.add_argument(
        "--visualizer_params_path",
        type=str,
        default="environments/slime_dph_5/config/env_visualizer-params.json",
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
    if check_args(args):
        print("Current args: ", args)
        main(args)
