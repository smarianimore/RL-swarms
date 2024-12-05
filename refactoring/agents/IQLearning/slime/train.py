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
    
    # Q-Learning
    # Q_table
    #qtable = {str(i): np.zeros([4, n_actions]) for i in range(population, population + learner_population)}
    qtable = np.zeros([learner_population, n_obs, n_actions])
    alpha = l_params["alpha"]  # DOC learning rate (0 learn nothing 1 learn suddenly)
    gamma = l_params["gamma"]  # DOC discount factor (0 care only bout immediate rewards, 1 care only about future ones)
    epsilon = l_params["epsilon"]  # DOC chance of random action
    epsilon_min = l_params["epsilon_min"]  # DOC chance of random action
    decay_type = l_params["decay_type"]  # DOC di quanto diminuisce epsilon ogni episode (e.g. 1500 episodes => decay = 0.9995)
    decay = l_params["decay"]  # DOC di quanto diminuisce epsilon ogni episode (e.g. 1500 episodes => decay = 0.9995)
    train_episodes = l_params["train_episodes"]

    # DOC dict che tiene conto della frequenza di scelta delle action per ogni episodio {episode: {action: _, action: _, ...}}
    actions_dict = {
        str(ep): {
            str(ac): 0 
            for ac in range(n_actions)
        } for ep in range(1, train_episodes + 1)
    }  # DOC 0 = walk, 1 = lay_pheromone, 2 = follow_pheromone
    # DOC dict che tiene conto della frequenza di scelta delle action di ogni agent per ogni episodio {episode: {agent: {action: _, action: _, ...}}}
    action_dict = {
        str(ep): {
            str(ag): {
                str(ac): 0 
                for ac in range(n_actions)
            } for ag in range(population, population + learner_population)
        } for ep in range(1, train_episodes + 1)
    }
    # DOC dict che tiene conto della reward di ogni agente per ogni episodio {episode: {agent: _}}
    reward_dict = {
        str(ep): {
            str(ag): 0 
            for ag in range(population, population + learner_population)
        }
        for ep in range(1, train_episodes + 1)
    }
    avg_reward_dict = {}
    # DOC dict che tiene conto dela dimensioni di ogni cluster per ogni episodio
    cluster_dict = {}
    
    return (
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
        avg_reward_dict,
        cluster_dict
    )

def create_logger(curdir, params, l_params, log_params):
    from logger import Logger
    train_log_every = log_params["train_log_every"]
    deep_algo = log_params["deep_algorithm"]
    buffer_size = log_params["buffer_size"]
    log =  Logger(curdir, params, l_params, log_params, train=True, deep_algo=deep_algo, buffer_size=buffer_size)
    return log, train_log_every

def train(
        env, 
        params:dict, 
        qtable, 
        actions_dict:dict, 
        action_dict:dict, 
        reward_dict:dict, 
        avg_reward_dict:dict,
        cluster_dict:dict,
        train_episodes:int, 
        train_log_every, 
        alpha:float, 
        gamma:float, 
        decay_type:str,
        decay:float,
        epsilon:float,
        epsilon_min:float,
        #output_file,
        logger,
        visualizer=None
    ):
    # TRAINING
    print("Start training...\n")
    
    n_actions = env.actions_n()
    old_s = {}  # DOC old state for each agent {agent: old_state}
    old_a = {}

    for ep in tqdm(range(1, train_episodes + 1), desc="EPISODES", colour='red', position=0, leave=False):
        env.reset()
        
        for tick in tqdm(range(1, params['episode_ticks'] + 1), desc="TICKS", colour='green', position=1, leave=False):
            for agent in env.agent_iter(max_iter=params['learner_population']):
                cur_state, reward, _, _, _ = env.last(agent)
                #cur_s = utils.state_to_int_map(cur_state)
                cur_s = env.convert_observation(cur_state)
                
                if ep == 1 and tick == 1:
                    #action = env.action_space(agent).sample()
                    action = np.random.randint(0, n_actions)
                else:
                    #old_value = qtable[int(agent), old_s[agent], action]
                    old_value = qtable[int(agent), old_s[agent], old_a[agent]]
                    next_max = np.max(qtable[int(agent), cur_s])  # QUESTION: was with [action] too
                    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                    qtable[int(agent), old_s[agent], old_a[agent]] = new_value
                    #qtable[int(agent), old_s[agent], action] = new_value
                    if random.uniform(0, 1) < epsilon:
                        action = np.random.randint(0, n_actions)
                        #action = env.action_space(agent).sample()
                    else:
                        action = np.argmax(qtable[int(agent)][cur_s])
                env.step(action)

                old_s[agent] = cur_s
                old_a[agent] = action

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
            #print(json.dumps(action_dict, indent=2))
        if decay_type == "log":
            #epsilon *= decay
            epsilon = max(epsilon * decay, epsilon_min)
        elif decay_type == "linear":
            epsilon = max(epsilon - (1 - decay), epsilon_min)
        
        #avg_rew = (sum(reward_dict[str(ep)].values()) / params["episode_ticks"]) / params["learner_population"]
        #avg_reward_dict[str(ep)] = round(avg_rew, 2) 
        #avg_cluster = env.avg_cluster()
        #cluster_dict[str(ep)] = round(avg_cluster, 2)
        #values = (ep, tick, epsilon, avg_cluster, avg_rew)        
        if ep % train_log_every == 0:
            avg_rew = round((sum(reward_dict[str(ep)].values()) / params["episode_ticks"]) / params["learner_population"], 2)
            #avg_cluster = round(env.avg_cluster(), 2)
            avg_cluster = round(env.avg_cluster2(), 2)
            eps = round(epsilon, 4)
            #metrics_dict["Episode"] = ep
            #metrics_dict["Tick"] = tick
            #metrics_dict["Epsilon"] = epsilon
            #metrics_dict["Avg cluster X episode"] = avg_cluster
            #metrics_dict["Avg reward X episode"] = avg_rew
            #metrics_dict.update(actions_dict[str(ep)])
            value = [ep, tick * ep, avg_cluster, avg_rew]
            value.extend(list(actions_dict[str(ep)].values()))
            value.append(eps)
            logger.load_value(value)
            #log_train(
            #    params,
            #    logger,
            #    env,
            #    ep,
            #    tick,
            #    epsilon,
            #    actions_dict,
            #    reward_dict,
            #    metrics_dict
            #)
            #log_train(
            #    ep,
            #    epsilon,
            #    cluster_dict,
            #    avg_reward_dict,
            #    actions_dict,
            #    action_dict,
            #    params,
            #    output_file
            #)

    #print(json.dumps(cluster_dict, indent=2))
    logger.empty_table()
    env.close()
    if visualizer != None:
        visualizer.close()
    print("Training finished!\n")

    return qtable

'''
def log_train(
        params,
        logger,
        env,
        ep,
        tick,
        epsilon,
        actions_dict,
        reward_dict,
        metrics_dict
    ):
    avg_rew = (sum(reward_dict[str(ep)].values()) / params["episode_ticks"]) / params["learner_population"]
    avg_cluster = env.avg_cluster()
    metrics_dict["Episode"] = ep
    metrics_dict["Tick"] = tick
    metrics_dict["Epsilon"] = epsilon
    metrics_dict["Avg cluster X episode"] = avg_cluster
    metrics_dict["Avg reward X episode"] = avg_rew
    metrics_dict.update(actions_dict[str(ep)])
    logger.load_values(metrics_dict)

def log_train(
        ep,
        epsilon,
        cluster_dict,
        avg_reward_dict,
        actions_dict,
        action_dict,
        params,
        output_file
    ):
    #print(f"\tepisode reward: {reward_episode}")
    # From NetlogoDataAnalysis: Episode, Tick, Avg cluster size X tick, move-toward-chemical (2), random-walk (0), drop-chemical (1), (learner 0)-move-toward-chemical, ..., Avg reward X episode
   
    # Actions:
    #   0: random-walk 
    #   1: drop-chemical 
    #   2: move-toward-chemical 
    #   3: move-away-chemical 
    #   4: walk-and-drop 
    #   5: move-and-drop

    avg_cluster = round(sum(cluster_dict.values()) / len(cluster_dict), 2)
    avg_reward = round(sum(avg_reward_dict.values()) / len(avg_reward_dict), 2)
    if ep > 1:
        std_cluster = round(statistics.stdev(cluster_dict.values()), 2)
        std_reward = round(statistics.stdev(avg_reward_dict.values()), 2)
    else:
        std_cluster = 0.0
        std_reward = 0.0
    min_cluster = min(cluster_dict.values())
    max_cluster = max(cluster_dict.values())
    min_reward = min(avg_reward_dict.values())
    max_reward = max(avg_reward_dict.values())

    with open(output_file, 'a') as f:
        f.write(
            ("{}, {}, {}, {}, {}, {}, {}, {}, ").format(
                ep,
                params['episode_ticks'] * ep, 
                round(epsilon, 2),
                cluster_dict[str(ep)],
                avg_cluster,
                std_cluster,
                min_cluster,
                max_cluster
            )
        )
        
        for v in actions_dict[str(ep)].values():
            f.write(f"{v}, ")

        #for l in range(params['population'], params['population'] + params['learner_population']):
        #    for v in action_dict[str(ep)][str(l)].values():
        #        f.write(f"{v}, ")
        
        #f.write(f"{avg_reward_dict[str(ep)]}\n")
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
    print(f"\tEpsilon: {round(epsilon, 2)}")
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
'''

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
        avg_reward_dict,
        cluster_dict
    ) = create_agent(params, l_params, n_obs, n_actions)
    logger, train_log_every = create_logger(curdir, params, l_params, log_params)

    train_start = datetime.datetime.now()
    qtable = train(
        env,
        params,
        qtable,
        actions_dict,
        action_dict,
        reward_dict,
        avg_reward_dict,
        cluster_dict,
        train_episodes,
        train_log_every,
        alpha,
        gamma,
        decay_type,
        decay,
        epsilon,
        epsilon_min,
        #output_file,
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
