import random
import numpy as np
from tqdm import tqdm

def policy(env, state, agent, threshold):
    cluster = env._compute_cluster(int(agent)) 
    #if cluster > 5:
        #action = 1
        #else:
    if np.all(state > threshold):
        action = 5
    else:
        action = 4
    
    return action

def run(env, params, episodes, visualizer=None):
    population = params['population']
    learner_population = params['learner_population']
    reward_dict = {
        str(ep): {
            str(ag): 0 
            for ag in range(population, population + learner_population)
        }
        for ep in range(1, episodes + 1)
    }
    cluster_dict = {str(ep): 0.0 for ep in range(1, episodes + 1)}
    avg_reward_dict = []
    avg_cluster_dict = []

    print("Start running...\n")
        # follow_pheromone e se sei in un cluster lay_pheromone
    for ep in tqdm(range(1, episodes + 1), desc="EPISODES", colour='red', position=0, leave=False):
        env.reset()
        
        for tick in tqdm(range(1, params['episode_ticks'] + 1), desc="TICKS", colour='green', position=1, leave=False):
            for agent in env.agent_iter(max_iter=params['learner_population']):
                cur_state, reward, _, _, _ = env.last(agent)
                action = policy(env, cur_state, agent, env.sniff_threshold)
                env.step(action)
            
                reward_dict[str(ep)][str(agent)] += round(reward, 2)
            
            cluster_dict[str(ep)] += round(env.avg_cluster2(), 2) 
            if visualizer != None:
                visualizer.render(
                    env.patches,
                    env.learners,
                    env.turtles,
                    env.fov,
                    env.ph_fov
                )
        
        avg_rew = round((sum(reward_dict[str(ep)].values()) / params["episode_ticks"]) / params["learner_population"], 2)
        avg_reward_dict.append(avg_rew)
        avg_cluster = round(cluster_dict[str(ep)] / params["episode_ticks"], 2)
        avg_cluster_dict.append(avg_cluster)

    env.close()
    if visualizer != None:
        visualizer.close()
    print("Training finished!\n")

    return avg_reward_dict, avg_cluster_dict