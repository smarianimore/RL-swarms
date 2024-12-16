import random

from tqdm import tqdm

def policy(epsilon):
    # Follow pheromone (action 2)
    # and some time drop pheromone (action 1)  
    rnd = random.uniform(0, 1) 
    if rnd < epsilon:
        action = 1
    else:
        action = 2
    return action

def run(env, params, episodes, epsilon, visualizer=None):
    population = params['population']
    learner_population = params['learner_population']
    reward_dict = {
        str(ep): {
            str(ag): 0 
            for ag in range(population, population + learner_population)
        }
        for ep in range(1, episodes + 1)
    }
    avg_reward_dict = []
    avg_cluster_dict = []

    print("Start running...\n")
        # follow_pheromone e se sei in un cluster lay_pheromone
    for ep in tqdm(range(1, episodes + 1), desc="EPISODES", colour='red', position=0, leave=False):
        env.reset()
        
        for tick in tqdm(range(1, params['episode_ticks'] + 1), desc="TICKS", colour='green', position=1, leave=False):
            for agent in env.agent_iter(max_iter=params['learner_population']):
                cur_state, reward, _, _, _ = env.last(agent)
                action = policy(epsilon)
                env.step(action)
            
                reward_dict[str(ep)][str(agent)] += round(reward, 2)
            
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
        avg_cluster = round(env.avg_cluster2(), 2)
        avg_cluster_dict.append(avg_cluster)

    env.close()
    if visualizer != None:
        visualizer.close()
    print("Training finished!\n")

    return avg_reward_dict, avg_cluster_dict