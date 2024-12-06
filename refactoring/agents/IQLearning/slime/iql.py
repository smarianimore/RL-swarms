
import random
import numpy as np
from tqdm import tqdm

def train(
        env, 
        params:dict, 
        qtable, 
        actions_dict:dict, 
        action_dict:dict, 
        reward_dict:dict, 
        train_episodes:int, 
        train_log_every, 
        alpha:float, 
        gamma:float, 
        decay_type:str,
        decay:float,
        epsilon:float,
        epsilon_min:float,
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
                cur_s = env.convert_observation(cur_state)
                
                if ep == 1 and tick == 1:
                    #action = env.action_space(agent).sample()
                    action = np.random.randint(0, n_actions)
                else:
                    old_value = qtable[int(agent), old_s[agent], old_a[agent]]
                    next_max = np.max(qtable[int(agent), cur_s])  # QUESTION: was with [action] too
                    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                    qtable[int(agent), old_s[agent], old_a[agent]] = new_value
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
        
        if decay_type == "log":
            epsilon = max(epsilon * decay, epsilon_min)
        elif decay_type == "linear":
            epsilon = max(epsilon - (1 - decay), epsilon_min)
        
        if ep % train_log_every == 0:
            avg_rew = round((sum(reward_dict[str(ep)].values()) / params["episode_ticks"]) / params["learner_population"], 2)
            #avg_cluster = round(env.avg_cluster(), 2)
            avg_cluster = round(env.avg_cluster2(), 2)
            eps = round(epsilon, 4)
            value = [ep, tick * ep, avg_cluster, avg_rew]
            value.extend(list(actions_dict[str(ep)].values()))
            value.append(eps)
            logger.load_value(value)
            
            #print(f"\nEPISODE: {ep}")
            #print(f"\tEpsilon: {round(epsilon, 2)}")
            #print("\tCluster metrics up to now:")
            #print("\t  - avg cluster in this episode: ", cluster_dict[str(ep)])
            #print("\t  - avg cluster: ", avg_cluster)
            #print("\t  - avg cluster std: ", std_cluster)
            #print("\t  - min cluster: ", min_cluster)
            #print("\t  - max cluster: ", max_cluster)
            #print("\tReward metrics up to now:")
            #print("\t  - avg reward in this episode: ", avg_reward_dict[str(ep)])
            #print("\t  - avg reward: ", avg_reward)
            #print("\t  - avg reward std: ", std_reward)
            #print("\t  - min reward: ", min_reward)
            #print("\t  - max reward: ", max_reward)

    logger.empty_table()
    env.close()
    if visualizer != None:
        visualizer.close()
    print("Training finished!\n")

    return qtable

def eval(
        env,
        params:dict, 
        actions_dict,
        action_dict,
        reward_dict,
        test_episodes:int,
        qtable,
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
                s = env.convert_observation(state)
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
        
        if ep % test_log_every == 0:
            avg_rew = round((sum(reward_dict[str(ep)].values()) / params["episode_ticks"]) / params["learner_population"], 2)
            #avg_cluster = round(env.avg_cluster(), 2)
            avg_cluster = round(env.avg_cluster2(), 2)
            value = [ep, tick * ep, avg_cluster, avg_rew]
            value.extend(list(actions_dict[str(ep)].values()))
            logger.load_value(value)
    
    logger.empty_table()
    env.close()
    if visualizer != None:
        visualizer.close()
    print("Testing finished!")