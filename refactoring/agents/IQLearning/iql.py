import random
import numpy as np
from tqdm import tqdm

def create_agent(params: dict, l_params: dict, n_obs, n_actions, train):
    population = params['population']
    learner_population = params["cluster_learners"] + params["scatter_learners"]
    episodes =  l_params["train_episodes"] if train else l_params["test_episodes"]
    # DOC dict che tiene conto della frequenza di scelta delle action per ogni episodio {episode: {action: _, action: _, ...}}
    # Actions:
    #   0: random-walk 
    #   1: drop-chemical 
    #   2: move-toward-chemical 
    #   3: move-away-chemical 
    #   4: walk-and-drop 
    #   5: move-and-drop
    cluster_actions_dict = {
        str(ep): {
            str(ac): 0 
            for ac in range(n_actions)
        } for ep in range(1, episodes + 1)
    }  # DOC 0 = walk, 1 = lay_pheromone, 2 = follow_pheromone
    # DOC dict che tiene conto della frequenza di scelta delle action di ogni agent per ogni episodio {episode: {agent: {action: _, action: _, ...}}}
    cluster_action_dict = {
        str(ep): {
            str(ag): {
                str(ac): 0 
                for ac in range(n_actions)
            } for ag in range(population, population + params["cluster_learners"])
        } for ep in range(1, episodes + 1)
    }
    # DOC dict che tiene conto della reward di ogni agente per ogni episodio {episode: {agent: _}}
    cluster_reward_dict = {
        str(ep): {
            str(ag): 0 
            for ag in range(population, population + params["cluster_learners"])
        }
        for ep in range(1, episodes + 1)
    }
    
    scatter_actions_dict = {
        str(ep): {
            str(ac): 0 
            for ac in range(n_actions)
        } for ep in range(1, episodes + 1)
    }  # DOC 0 = walk, 1 = lay_pheromone, 2 = follow_pheromone
    # DOC dict che tiene conto della frequenza di scelta delle action di ogni agent per ogni episodio {episode: {agent: {action: _, action: _, ...}}}
    scatter_action_dict = {
        str(ep): {
            str(ag): {
                str(ac): 0 
                for ac in range(n_actions)
            } for ag in range(population + params["cluster_learners"], learner_population)
        } for ep in range(1, episodes + 1)
    }
    # DOC dict che tiene conto della reward di ogni agente per ogni episodio {episode: {agent: _}}
    scatter_reward_dict = {
        str(ep): {
            str(ag): 0 
            for ag in range(population + params["cluster_learners"], learner_population)
        }
        for ep in range(1, episodes + 1)
    }

    cluster_dict = {str(ep): 0.0 for ep in range(1, episodes + 1)}

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
            cluster_dict,
            cluster_actions_dict,
            cluster_action_dict,
            cluster_reward_dict,
            scatter_actions_dict,
            scatter_action_dict,
            scatter_reward_dict,
        )
    else:
        return (
            episodes,
            cluster_dict,
            cluster_actions_dict,
            cluster_action_dict,
            cluster_reward_dict,
            scatter_actions_dict,
            scatter_action_dict,
            scatter_reward_dict,
        )

def train(
        env, 
        params:dict, 
        qtable, 
        cluster_dict,
        cluster_actions_dict,
        cluster_action_dict,
        cluster_reward_dict,
        scatter_actions_dict,
        scatter_action_dict,
        scatter_reward_dict,
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
    
    n_actions = env.actions_n()
    old_s = {}  # DOC old state for each agent {agent: old_state}
    old_a = {}
    actions = [4, 5]
    scatter_actions = np.array([0, 1, 3])
    best_cluster_reward = 0.0
    AGENTS_NUM = env.cluster_learners + env.scatter_learners
    only_cluster_dict = {str(ep): 0.0 for ep in range(1, train_episodes + 1)}
    mixed_cluster_dict = {str(ep): 0.0 for ep in range(1, train_episodes + 1)}
    only_scatter_dict = {str(ep): 0.0 for ep in range(1, train_episodes + 1)}
    mixed_scatter_dict = {str(ep): 0.0 for ep in range(1, train_episodes + 1)}

    # TRAINING
    print("Start training...\n")

    for ep in tqdm(range(1, train_episodes + 1), desc="EPISODES", colour='red', position=0, leave=False):
        env.reset()
        
        for tick in tqdm(range(1, params['episode_ticks'] + 1), desc="TICKS", colour='green', position=1, leave=False):
            for agent in env.agent_iter(max_iter=AGENTS_NUM):
                cur_state, reward, _, _, _ = env.last(agent)
                cur_s = env.convert_observation2(cur_state)

                if ep == 1 and tick == 1:
                    #action = env.action_space(agent).sample()
                    action = np.random.randint(0, n_actions)
                else:
                    # QTable update
                    old_value = qtable[int(agent), old_s[agent], old_a[agent]]
                    next_max = np.max(qtable[int(agent), cur_s])  # QUESTION: was with [action] too
                    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                    qtable[int(agent), old_s[agent], old_a[agent]] = new_value
                    
                    # next_action
                    if random.uniform(0, 1) < epsilon:
                        action = np.random.randint(0, n_actions)
                        #action = env.action_space(agent).sample()
                    else:
                        action = np.argmax(qtable[int(agent)][cur_s])

                #print(actions[action])
                #breakpoint()
                #env.step(actions[action])
                if env.learners[int(agent)]["mode"] == 's':
                    env.step(scatter_actions[action].item())
                else:
                    env.step(action)

                old_s[agent] = cur_s
                old_a[agent] = action

                if env.learners[int(agent)]["mode"] == 'c':
                    cluster_actions_dict[str(ep)][str(action)] += 1
                    #cluster_action_dict[str(ep)][str(agent)][str(action)] += 1
                    cluster_reward_dict[str(ep)][str(agent)] += round(reward, 2)
                elif env.learners[int(agent)]["mode"] == 's': 
                    scatter_actions_dict[str(ep)][str(action)] += 1
                    #scatter_action_dict[str(ep)][str(agent)][str(action)] += 1
                    scatter_reward_dict[str(ep)][str(agent)] += round(reward, 2)
                
            if env.cluster_learners == 0 or env.scatter_learners == 0:
                cluster_dict[str(ep)] += round(env.avg_cluster2(), 2) 
            else:
                (
                    avg_only_cluster,
                    avg_mixed_cluster,
                    avg_only_scatter,
                    avg_mixed_scatter
                ) = env.avg_cluster2()
                only_cluster_dict[str(ep)] += round(avg_only_cluster, 2)
                mixed_cluster_dict[str(ep)] += round(avg_mixed_cluster, 2)
                only_scatter_dict[str(ep)] += round(avg_only_scatter, 2)
                mixed_scatter_dict[str(ep)] += round(avg_mixed_scatter, 2)
            
            if visualizer != None:
                visualizer.render(
                    env.patches,
                    env.learners,
                    env.turtles,
                    env.fov,
                    env.ph_fov
                )
        
        if decay_type == "log":
            epsilon = max(epsilon * decay, epsilon_min)
        elif decay_type == "linear":
            epsilon = max(epsilon - (1 - decay), epsilon_min)
        
        if ep % train_log_every == 0:
            value = [ep, tick * ep]
            
            if env.cluster_learners == 0 or env.scatter_learners == 0:
                avg_cluster = round(cluster_dict[str(ep)] / params["episode_ticks"], 2)
                value.append(avg_cluster)
            else:
                avg_only_cluster = round(only_cluster_dict[str(ep)] / params["episode_ticks"], 2)
                value.append(avg_only_cluster)
                avg_mixed_cluster = round(mixed_cluster_dict[str(ep)] / params["episode_ticks"], 2)
                value.append(avg_mixed_cluster)
                avg_only_scatter = round(only_scatter_dict[str(ep)] / params["episode_ticks"], 2)
                value.append(avg_only_scatter)
                avg_mixed_scatter = round(mixed_scatter_dict[str(ep)] / params["episode_ticks"], 2)
                value.append(avg_mixed_scatter)
            
            if params["cluster_learners"] > 0:
                cluster_avg_rew = round((sum(cluster_reward_dict[str(ep)].values()) / params["episode_ticks"]) / params["cluster_learners"], 4)
            else:
                cluster_avg_rew = 0.0
            value.append(cluster_avg_rew)
            value.extend(list(cluster_actions_dict[str(ep)].values()))

            if params["scatter_learners"] > 0:
                scatter_avg_rew = round((sum(scatter_reward_dict[str(ep)].values()) / params["episode_ticks"]) / params["scatter_learners"], 4)
            else:
                scatter_avg_rew = 0.0
            value.append(scatter_avg_rew)
            value.extend(list(scatter_actions_dict[str(ep)].values()))
            
            eps = round(epsilon, 4)
            value.append(eps)
            
            logger.load_value(value)

            if best_cluster_reward < cluster_avg_rew:
                print("\nMetrics ")
                if env.cluster_learners == 0 or env.scatter_learners == 0:
                    print(" - cluster: ", avg_cluster)
                else:
                    print(" - only_cluster: ", avg_only_cluster)
                    print(" - mixed_cluster: ", avg_mixed_cluster)
                    print(" - only_scatter: ", avg_only_scatter)
                    print(" - mixed_scatter: ", avg_mixed_scatter)
                print(" - cluster_reward: ", cluster_avg_rew)
                print(" - scatter_reward: ", scatter_avg_rew)
                print(" - epsilon: ", eps)
                best_cluster_reward = cluster_avg_rew

    logger.empty_table()
    env.close()
    if visualizer != None:
        visualizer.close()
    print("Training finished!\n")

    return qtable

def eval(
        env,
        params:dict, 
        cluster_dict,
        cluster_actions_dict,
        cluster_action_dict,
        cluster_reward_dict,
        scatter_actions_dict,
        scatter_action_dict,
        scatter_reward_dict,
        test_episodes:int,
        qtable,
        test_log_every:int,
        logger,
        visualizer=None
    ):
    # DOC Evaluate agent's performance after Q-learning
    #n_actions = env.actions_n()
    actions = [4, 5]
    scatter_actions = np.array([0, 1, 3])
    AGENTS_NUM = env.cluster_learners + env.scatter_learners
    only_cluster_dict = {str(ep): 0.0 for ep in range(1, test_episodes + 1)}
    mixed_cluster_dict = {str(ep): 0.0 for ep in range(1, test_episodes + 1)}
    only_scatter_dict = {str(ep): 0.0 for ep in range(1, test_episodes + 1)}
    mixed_scatter_dict = {str(ep): 0.0 for ep in range(1, test_episodes + 1)}
    
    print("Start testing...\n")
    
    for ep in tqdm(range(1, test_episodes + 1), desc="EPISODES", colour='red', leave=False):
        env.reset()
        for tick in tqdm(range(1, params['episode_ticks'] + 1), desc="TICKS", colour='green', leave=False):
            for agent in env.agent_iter(max_iter=AGENTS_NUM):
                state, reward, _, _, _ = env.last(agent)
                s = env.convert_observation2(state)
                action = np.argmax(qtable[int(agent)][s])
                
                #env.step(action)
                #print(actions[action])
                #breakpoint()
                #env.step(actions[action])
                if env.learners[int(agent)]["mode"] == 's':
                    env.step(scatter_actions[action].item())
                else:
                    env.step(action)
                
                if env.learners[int(agent)]["mode"] == 'c':
                    cluster_actions_dict[str(ep)][str(action)] += 1
                    #cluster_action_dict[str(ep)][str(agent)][str(action)] += 1
                    cluster_reward_dict[str(ep)][str(agent)] += round(reward, 2)
                elif env.learners[int(agent)]["mode"] == 's': 
                    scatter_actions_dict[str(ep)][str(action)] += 1
                    #scatter_action_dict[str(ep)][str(agent)][str(action)] += 1
                    scatter_reward_dict[str(ep)][str(agent)] += round(reward, 2)
            
            if env.cluster_learners == 0 or env.scatter_learners == 0:
                cluster_dict[str(ep)] += round(env.avg_cluster2(), 2) 
            else:
                (
                    avg_only_cluster,
                    avg_mixed_cluster,
                    avg_only_scatter,
                    avg_mixed_scatter
                ) = env.avg_cluster2()
                only_cluster_dict[str(ep)] += round(avg_only_cluster, 2)
                mixed_cluster_dict[str(ep)] += round(avg_mixed_cluster, 2)
                only_scatter_dict[str(ep)] += round(avg_only_scatter, 2)
                mixed_scatter_dict[str(ep)] += round(avg_mixed_scatter, 2)

            if visualizer != None:
                visualizer.render(
                    env.patches,
                    env.learners,
                    env.fov,
                    env.ph_fov
                )
        
        if ep % test_log_every == 0:
            value = [ep, tick * ep]
            
            if env.cluster_learners == 0 or env.scatter_learners == 0:
                avg_cluster = round(cluster_dict[str(ep)] / params["episode_ticks"], 2)
                value.append(avg_cluster)
            else:
                avg_only_cluster = round(only_cluster_dict[str(ep)] / params["episode_ticks"], 2)
                value.append(avg_only_cluster)
                avg_mixed_cluster = round(mixed_cluster_dict[str(ep)] / params["episode_ticks"], 2)
                value.append(avg_mixed_cluster)
                avg_only_scatter = round(only_scatter_dict[str(ep)] / params["episode_ticks"], 2)
                value.append(avg_only_scatter)
                avg_mixed_scatter = round(mixed_scatter_dict[str(ep)] / params["episode_ticks"], 2)
                value.append(avg_mixed_scatter)
            
            if params["cluster_learners"] > 0:
                cluster_avg_rew = round((sum(cluster_reward_dict[str(ep)].values()) / params["episode_ticks"]) / params["cluster_learners"], 4)
            else:
                cluster_avg_rew = 0.0
            value.append(cluster_avg_rew)
            value.extend(list(cluster_actions_dict[str(ep)].values()))

            if params["scatter_learners"] > 0:
                scatter_avg_rew = round((sum(scatter_reward_dict[str(ep)].values()) / params["episode_ticks"]) / params["scatter_learners"], 4)
            else:
                scatter_avg_rew = 0.0
            value.append(scatter_avg_rew)
            value.extend(list(scatter_actions_dict[str(ep)].values()))
            
            logger.load_value(value)
    
    logger.empty_table()
    env.close()
    if visualizer != None:
        visualizer.close()
    print("Testing finished!")