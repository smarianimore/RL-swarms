import time
from tqdm import tqdm

from collections import deque
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ReplayMemory:
    def __init__(self, max_len=100000):
        self.memory = deque([], maxlen=max_len)

    def append(self, expirience):
        self.memory.append(expirience)

    def sample(self, size):
        return random.sample(self.memory, size)

    def __len__(self):
        return len(self.memory)

class IDQN:
    def __init__(self, env, visualizer, logger, use_gpu, seed, **kwargs):
        self.env = env
        self.vis = visualizer
        self.logger = logger
        self.replay_memory_size = kwargs["replay_memory_size"]
        self.mini_bach_size = kwargs["mini_bach_size"]
        self.policy_sync_rate = kwargs["policy_sync_rate"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.epsilon_init = kwargs["epsilon_init"]
        self.epsilon_min = kwargs["epsilon_min"]
        self.decay = kwargs["decay"]
        self.decay_type = kwargs["decay_type"]
        self.learning_rate_a = kwargs["learning_rate_a"]
        self.discount_factor_g = kwargs["discount_factor_g"]

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        #torch.use_deterministic_algorithms(True)
        if use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.device == "cuda":
                torch.backends.cudnn.deterministic = True 
        else:
            self.device = torch.device("cpu")

        self.obs_dim = env.observations_n()
        self.actions_dim = env.actions_n()

        self.policy_nets = {str(l): Net(self.obs_dim, self.actions_dim, hidden_dim=self.hidden_dim).to(self.device) for l in env.learners}
        self.target_policy_nets = {str(l): Net(self.obs_dim, self.actions_dim, hidden_dim=self.hidden_dim).to(self.device) for l in env.learners}
        self.optimizers = {str(l): torch.optim.Adam(self.policy_nets[str(l)].parameters(), lr=self.learning_rate_a) for l in env.learners}
        self.loss_fn = nn.MSELoss()
    
    def _init_target_policies(self):
        for t in self.target_policy_nets:
            self.target_policy_nets[t].load_state_dict(self.policy_nets[t].state_dict())
    
    def _get_log_data_structs(self, episodes, train, **params):
        population = params['population']
        learner_population = params["cluster_learners"] + params["scatter_learners"]

        cluster_actions_dict = {
            str(ep): {
                str(ac): 0 
                for ac in range(self.actions_dim)
            } for ep in range(1, episodes + 1)
        }  # DOC 0 = walk, 1 = lay_pheromone, 2 = follow_pheromone
        # DOC dict che tiene conto della frequenza di scelta delle action di ogni agent per ogni episodio {episode: {agent: {action: _, action: _, ...}}}
        cluster_action_dict = {
            str(ep): {
                str(ag): {
                    str(ac): 0 
                    for ac in range(self.actions_dim)
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
                for ac in range(self.actions_dim)
            } for ep in range(1, episodes + 1)
        }  # DOC 0 = walk, 1 = lay_pheromone, 2 = follow_pheromone
        # DOC dict che tiene conto della frequenza di scelta delle action di ogni agent per ogni episodio {episode: {agent: {action: _, action: _, ...}}}
        scatter_action_dict = {
            str(ep): {
                str(ag): {
                    str(ac): 0 
                    for ac in range(self.actions_dim)
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
            loss_dict = {str(ep): {str(ag): 0 for ag in range(population, population + learner_population)} for ep in range(1, episodes + 1)}
            return (
                cluster_dict,
                cluster_actions_dict,
                cluster_action_dict,
                cluster_reward_dict,
                scatter_actions_dict,
                scatter_action_dict,
                scatter_reward_dict,
                loss_dict
            )
        else:
            return (
                cluster_dict,
                cluster_actions_dict,
                cluster_action_dict,
                cluster_reward_dict,
                scatter_actions_dict,
                scatter_action_dict,
                scatter_reward_dict,
            )


    def _select_action(self, agent, obs, epsilon):
        if random.random() < epsilon:
            action = np.random.randint(0, self.actions_dim)
            action = torch.tensor(action, dtype=torch.int64, device=self.device)
        else:
            with torch.no_grad():
                action = self.policy_nets[agent](obs.unsqueeze(dim=0)).squeeze().argmax()
        return action
    
    def _optimize(self, agent, mini_batch):
        obs, actions, new_obs, rewards, dones = zip(*mini_batch)
        obs = torch.stack(obs)
        actions = torch.stack(actions)
        new_obs = torch.stack(new_obs)
        rewards = torch.stack(rewards)
        dones = torch.tensor(dones).float().to(self.device)

        with torch.no_grad():
            targets_q = rewards + (1 - dones) * self.discount_factor_g * self.target_policy_nets[agent](new_obs).max(dim=1)[0]
        currents_q = self.policy_nets[agent](obs).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        #breakpoint()
        loss = self.loss_fn(currents_q, targets_q)
        self.optimizers[agent].zero_grad()
        loss.backward()
        self.optimizers[agent].step()
        return loss.item()

    def train(self, train_episodes, train_log_every, **params):
        replay_memories = {str(l): ReplayMemory(self.replay_memory_size) for l in self.env.learners}
        self._init_target_policies()
        
        (
            cluster_dict,
            cluster_actions_dict,
            cluster_action_dict,
            cluster_reward_dict,
            scatter_actions_dict,
            scatter_action_dict,
            scatter_reward_dict,
            loss_dict
        ) = self._get_log_data_structs(train_episodes, True, **params)
        
        epsilon = self.epsilon_init
        actions = torch.tensor([4, 5])
        old_obs = {}
        old_actions = {}
        AGENTS_NUM = self.env.cluster_learners + self.env.scatter_learners

        for ep in tqdm(range(1, train_episodes + 1), desc="EPISODES", colour='red', position=0, leave=False):
            self.env.reset()
            
            for tick in tqdm(range(1, params['episode_ticks'] + 1), desc="TICKS", colour='green', position=1, leave=False):
                for agent in self.env.agent_iter(max_iter=AGENTS_NUM):
                    obs, reward, _ , _, _ = self.env.last(agent)
                    obs = torch.tensor(obs, dtype=torch.float, device=self.device)
                    reward = torch.tensor(reward, dtype=torch.float, device=self.device) 
                    done = True if tick == params["episode_ticks"] else False
                    
                    if ep == 1 and tick == 1:
                        #action = env.action_space(agent).sample()
                        action = np.random.randint(0, self.actions_dim)
                        action = torch.tensor(action, dtype=torch.int64, device=self.device)
                    else:
                        action = self._select_action(agent, obs, epsilon)

                        prev_obs = old_obs[agent]
                        prev_action = old_actions[agent]
                        replay_memories[agent].append((prev_obs, prev_action, obs, reward, done))
                        if len(replay_memories[agent]) > self.mini_bach_size:
                            mini_batch = replay_memories[agent].sample(self.mini_bach_size)
                            loss = self._optimize(agent, mini_batch)
                            loss_dict[str(ep)][str(agent)] += round(loss, 2)

                            if tick == self.policy_sync_rate:
                                self.target_policy_nets[agent].load_state_dict(self.policy_nets[agent].state_dict())

                    #self.env.step(actions[action].item())
                    self.env.step(action.item())

                    old_obs[agent] = obs
                    old_actions[agent] = action

                    if self.env.learners[int(agent)]["mode"] == 'c':
                        cluster_actions_dict[str(ep)][str(action.item())] += 1
                        #cluster_action_dict[str(ep)][str(agent)][str(action)] += 1
                        cluster_reward_dict[str(ep)][str(agent)] += reward.item()
                    elif self.env.learners[int(agent)]["mode"] == 's': 
                        scatter_actions_dict[str(ep)][str(action.item())] += 1
                        #scatter_action_dict[str(ep)][str(agent)][str(action)] += 1
                        scatter_reward_dict[str(ep)][str(agent)] += reward.item()

                cluster_dict[str(ep)] += round(self.env.avg_cluster2(), 2) 
                if self.vis != None:
                    self.vis.render(
                        self.env.patches,
                        self.env.learners,
                        self.env.turtles
                    )
            if len(replay_memories[agent]) > self.mini_bach_size:
                if self.decay_type == "log":
                    #epsilon *= decay
                    epsilon = max(epsilon * self.decay, self.epsilon_min)
                elif self.decay_type == "linear":
                    epsilon = max(epsilon - (1 - self.decay), self.epsilon_min)

            if ep % train_log_every == 0:
                avg_cluster = round(cluster_dict[str(ep)] / params["episode_ticks"], 2)
                value = [ep, tick * ep, avg_cluster]
                
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
                
                # Double loss??
                avg_loss = round((sum(loss_dict[str(ep)].values()) / params["episode_ticks"]) / AGENTS_NUM, 2)
                value.append(avg_loss)
                
                self.logger.load_value(value)

                print("\nMetrics ")
                print(" - cluster:", avg_cluster)
                print(" - cluster_reward: ", cluster_avg_rew)
                print(" - scatter_reward: ", scatter_avg_rew)
                print(" - epsilon: ", eps)
                print(" - loss: ", avg_loss)

        self.logger.empty_table()
        self.env.close()
        if self.vis != None:
            self.vis.close()
        print("Training finished!\n")
        
        model = {agent: self.policy_nets[agent].state_dict() for agent in self.policy_nets}
        return model
    
    def eval(self):
        pass
    