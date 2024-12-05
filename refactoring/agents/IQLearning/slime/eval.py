from slime import Slime
import utils

import sys
import os
from tqdm import tqdm
import datetime

import argparse

import json
import numpy as np
import random
import statistics

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

def create_agent(params: dict, l_params: dict, n_obs, n_actions):
    population = params['population']
    learner_population = params['learner_population']
    test_episodes = l_params["test_episodes"]

    # DOC dict che tiene conto della frequenza di scelta delle action per ogni episodio {episode: {action: _, action: _, ...}}
    actions_dict = {
        str(ep): {
            str(ac): 0 
            for ac in range(n_actions)
        } for ep in range(1, test_episodes + 1)
    }  # DOC 0 = walk, 1 = lay_pheromone, 2 = follow_pheromone
    # DOC dict che tiene conto della frequenza di scelta delle action di ogni agent per ogni episodio {episode: {agent: {action: _, action: _, ...}}}
    action_dict = {
        str(ep): {
            str(ag): {
                str(ac): 0 
                for ac in range(n_actions)
            } for ag in range(population, population + learner_population)
        } for ep in range(1, test_episodes + 1)
    }
    # DOC dict che tiene conto della reward di ogni agente per ogni episodio {episode: {agent: _}}
    reward_dict = {
        str(ep): {
            str(ag): 0 
            for ag in range(population, population + learner_population)
        }
        for ep in range(1, test_episodes + 1)
    }
    avg_reward_dict = {}
    # DOC dict che tiene conto dela dimensioni di ogni cluster per ogni episodio
    cluster_dict = {}
    
    return (
        test_episodes,
        actions_dict,
        action_dict,
        reward_dict,
        avg_reward_dict,
        cluster_dict,
    )

def create_logger(curdir, params, l_params, log_params, weights_path):
    from logger import Logger
    test_log_every = log_params["test_log_every"]
    deep_algo = log_params["deep_algorithm"]
    buffer_size = log_params["buffer_size"]
    log =  Logger(
        curdir,
        params,
        l_params,
        log_params,
        train=False,
        deep_algo=deep_algo,
        buffer_size=buffer_size,
        weights_file=weights_path
    )
    return log, test_log_every

def eval(
        env,
        params:dict, 
        actions_dict,
        action_dict,
        reward_dict,
        avg_reward_dict,
        cluster_dict,
        test_episodes:int,
        qtable,
        #eval_file,
        test_log_every:int,
        logger,
        visualizer=None
    ):
    # DOC Evaluate agent's performance after Q-learning
    #n_actions = env.actions_n()

    print("Start testing...\n")
    
    for ep in tqdm(range(1, test_episodes + 1), desc="EPISODES", colour='red', leave=False):
        env.reset()
        for tick in tqdm(range(1, params['episode_ticks'] + 1), desc="TICKS", colour='green', leave=False):
            for agent in env.agent_iter(max_iter=params['learner_population']):
                state, reward, _, _, _ = env.last(agent)
                #s = utils.state_to_int_map(state)
                s = env.convert_observation(state)

                #if random.uniform(0, 1) < epsilon:
                #    action = np.random.randint(0, n_actions)
                #    #action = env.action_space(agent).sample()
                #else:
                action = np.argmax(qtable[int(agent)][s])

                env.step(action)
                
                actions_dict[str(ep)][str(action)] += 1
                action_dict[str(ep)][str(agent)][str(action)] += 1
                reward_dict[str(ep)][str(agent)] += round(reward, 2)
            
            if visualizer != None:
                visualizer.render(
                    env.patches,
                    env.learners,
                    env.turtles
                )
            #env.move()
            #env._evaporate()
            #env._diffuse()
            #env.render()
            
        #avg_rew = (sum(reward_dict[str(ep)].values()) / params["episode_ticks"]) / params["learner_population"]
        #avg_reward_dict[str(ep)] = round(avg_rew, 2) 
        #cluster_dict[str(ep)] = round(env.avg_cluster(), 2)
        
        if ep % test_log_every == 0:
            #breakpoint()
            avg_rew = round((sum(reward_dict[str(ep)].values()) / params["episode_ticks"]) / params["learner_population"], 2)
            #avg_cluster = round(env.avg_cluster(), 2)
            avg_cluster = round(env.avg_cluster2(), 2)
            value = [ep, tick * ep, avg_cluster, avg_rew]
            value.extend(list(actions_dict[str(ep)].values()))
            logger.load_value(value)
            #log_eval(
            #    ep,
            #    actions_dict,
            #    action_dict,
            #    cluster_dict,
            #    avg_reward_dict,
            #    params,
            #    eval_file
            #)
    logger.empty_table()
    env.close()
    if visualizer != None:
        visualizer.close()
    print("Testing finished!")

def log_eval(ep, actions_dict, action_dict, cluster_dict, avg_reward_dict, params, eval_file):
    avg_cluster = round(sum(cluster_dict.values()) / len(cluster_dict), 2)
    min_cluster = min(cluster_dict.values())
    max_cluster = max(cluster_dict.values())
    avg_reward = round(sum(avg_reward_dict.values()) / len(avg_reward_dict), 2)
    min_reward = min(avg_reward_dict.values())
    max_reward = max(avg_reward_dict.values())
    if ep > 1:
        std_cluster = round(statistics.stdev(cluster_dict.values()), 2)
        std_reward = round(statistics.stdev(avg_reward_dict.values()), 2)
    else:
        std_cluster = 0.0
        std_reward = 0.0

    with open(eval_file, 'a') as f:
        f.write(
            ("{}, {}, {}, {}, {}, {}, {}, ").format(
                ep,
                params['episode_ticks'] * ep, 
                cluster_dict[str(ep)],
                avg_cluster,
                std_cluster,
                min_cluster,
                max_cluster
            )
        )
        
        for v in actions_dict[str(ep)].values():
            f.write(f"{v}, ")

        for l in range(params['population'], params['population'] + params['learner_population']):
            for v in action_dict[str(ep)][str(l)].values():
                f.write(f"{v}, ")
        
        f.write(
            ("{}, {}, {}, {}, {}\n").format(
                avg_reward_dict[str(ep)],
                avg_reward,
                std_reward,
                min_reward,
                max_reward
            )
        )

    print(f"\nEPISODE: {ep}")
    print("\tCluster metrics up to now:")
    print("\t  - avg cluster in this episode: ", cluster_dict[str(ep)])
    print("\t  - avg cluster: ", avg_cluster)
    print("\t  - avg cluster std: ", std_cluster)
    print("\t  - min cluster: ", min_cluster)
    print("\t  - max cluster: ", max_cluster)
    print("\tReward metrics up to now:")
    print("\t  - avg reward in this episode: ", avg_reward_dict[str(ep)])
    print("\t  - avg reward: ", avg_reward)
    print("\t  - avg reward std: ", std_reward)
    print("\t  - min reward: ", min_reward)
    print("\t  - max reward: ", max_reward)

def main(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    curdir = os.path.dirname(os.path.abspath(__file__))

    params, l_params, v_params, log_params = utils.read_params(
        args.params_path,
        args.learning_params_path,
        args.visualizer_params_path,
        args.logger_params_path
    )

    env = Slime(args.random_seed, **params)
    if args.render:
        from slime import SlimeVisualizer
        env_vis = SlimeVisualizer(env.W_pixels, env.H_pixels, **v_params)
    else:
        env_vis = None

    n_obs = env.observations_n()
    n_actions = env.actions_n()

    (
        test_episodes,
        actions_dict,
        action_dict,
        reward_dict,
        avg_reward_dict,
        cluster_dict,
    ) = create_agent(params, l_params, n_obs, n_actions)
    logger, test_log_every = create_logger(curdir, params, l_params, log_params, args.qtable_path)

    print("Loading Q-Table weights...")
    qtable = logger.load_model()
    #qtable = np.load(weights_file)
    print("Weights are loaded.\n")
            
    test_start = datetime.datetime.now()
    eval(
        env,
        params,
        actions_dict,
        action_dict,
        reward_dict,
        avg_reward_dict,
        cluster_dict,
        test_episodes,
        qtable,
        #output_file,
        test_log_every,
        logger,
        env_vis
    )
    test_end = datetime.datetime.now()
    logger.save_computation_time(test_end - test_start, train=False)
    print(f"\nTesting time: {test_end - test_start}")

def check_args(args):
    assert(
        args.params_path != ""
        and os.path.isfile(args.params_path)
        and args.params_path.endswith(".json")
    ), "[ERROR] params path is empty or is not a file or is not a json file"
    
    assert(
        args.learning_params_path != ""
        and os.path.isfile(args.learning_params_path)
        and args.learning_params_path.endswith(".json")
    ), "[ERROR] learning params path is empty or is not a file or is not a json file"
    
    assert(
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
        default="configs/qLearning-env-params.json",
        required=False
    )
    
    parser.add_argument(
        "--visualizer_params_path",
        type=str,
        default="configs/qLearning-visualizer-params.json",
        required=False
    )
    
    parser.add_argument(
        "--learning_params_path",
        type=str,
        default="configs/qLearning-learning-params.json",
        required=False
    )
    
    parser.add_argument(
        "--logger_params_path",
        type=str,
        default="configs/qLearning-logger-params.json",
        required=False
    )
    
    parser.add_argument(
        "--qtable_path",
        type=str,
        #default="qLearning-learning-params.json",
        required=False
    )
    
    parser.add_argument("--random_seed", type=int, default=42, required=False)
    
    parser.add_argument("--render", type=bool, default=False, required=False)

    args = parser.parse_args()
    if check_args(args):
         main(args)
