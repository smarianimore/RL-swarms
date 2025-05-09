import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np

from collections import deque, namedtuple
import random
import math

from tqdm import tqdm

class ReplayMemory(object):

    def __init__(self, Transition, capacity):
        self.memory = deque([], maxlen=capacity)
        self.Transition = Transition

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, epsilon):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        
        self.layer3 = nn.Linear(128, 256)
        self.layer4 = nn.Linear(256, 256)
        
        self.layer5 = nn.Linear(256, 512)
        self.layer6 = nn.Linear(512, n_actions)
        
        self.dropout = nn.Dropout(p=0.3)
        self.epsilon = epsilon


    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(self.dropout(x))
        x = F.relu(self.layer2(x) + x)
        
        x = self.layer3(x)
        x = F.relu(self.dropout(x))
        x = F.relu(self.layer4(x) + x)
        
        x = self.layer5(x)
        x = F.relu(self.dropout(x))
        return self.layer6(x)
    


def optimize_model(Transition, memory, policy_net, target_net, gamma, batch_size, device):
    if len(memory) < batch_size:
        return policy_net, target_net, None
    
    breakpoint() 
    transitions = memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    return policy_net, target_net, loss


def select_action(env, agent, actions_dim, state, steps_done, policy_net, device, epsilon_min, decay):
    sample = random.random()
    policy_net.epsilon = epsilon_min + (policy_net.epsilon - epsilon_min) * math.exp(-1. * steps_done * decay)
    
    if sample > policy_net.epsilon:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            breakpoint()
            return policy_net(state).max(1)[1].view(1, 1), policy_net
    else:
        action = np.random.randint(0, actions_dim)
        return action, policy_net
        #return torch.tensor([[action]], device=device, dtype=torch.long), policy_net
        #return torch.tensor([[env.action_space(agent).sample()]], device=device, dtype=torch.long), policy_net

def train(
        env, 
        params, 
        l_params, 
        train_episodes,
        train_log_every,
        normalize,
        logger,
        seed,
        visualizer=None
    ):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.backends.cudnn.deterministic = True 

    obs_dim = env.observations_n()
    actions_dim = env.actions_n()
    population = params['population']
    learner_population = params['learner_population']
    batch_size = l_params["batch_size"]
    learning_rate = l_params["lr"]
    epsilon = l_params["epsilon"]
    epsilon_min = l_params["epsilon_min"]
    alpha = l_params["alpha"]
    gamma = l_params["gamma"]
    decay = l_params["decay"]
    decay_type = l_params["decay_type"]
    update_net_every = l_params['update_net_every']
    memory_capacity = l_params["memory_capacity"]

    policy_nets = {ag: DQN(obs_dim, actions_dim, epsilon).to(device) for ag in range(population, population + learner_population)}
    target_nets = {ag: DQN(obs_dim, actions_dim, epsilon).to(device) for ag in range(population, population + learner_population)}
    optimizers = {i: optim.AdamW(policy_nets[i].parameters(), lr=learning_rate, amsgrad=True) for i in range(params['learner_population'])}
    schedulers = {i: StepLR(optimizers[i], step_size=1, gamma=l_params["step_lr"]) for i in range(params['learner_population'])}
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    memory = {i: ReplayMemory(Transition, memory_capacity) for i in range(params['learner_population'])}
    
    old_s = {}
    old_a = {}
    actions = [4, 5]
    cluster_dict = {}
    
    actions_dict = {str(ep): {str(ac): 0 for ac in range(actions_dim)} for ep in range(1, train_episodes + 1)}  # DOC 0 = walk, 1 = lay_pheromone, 2 = follow_pheromone
    #action_dict = {str(ep): {str(ag): {str(ac): 0 for ac in range(n_actions)} for ag in range(population, population + learner_population)} for ep in range(1, train_episodes + 1)}
    reward_dict = {str(ep): {str(ag): 0 for ag in range(population, population + learner_population)} for ep in range(1, train_episodes + 1)}
    epsilon = 0
    cur_lr = 0

    max_possible_reward = (((params['episode_ticks'] - 150)/params['episode_ticks']) * params['rew']) + \
        ((params['learner_population'] / params["cluster_threshold"]) * (params['rew'] ** 2))
    max_possible_pherormone = env.lay_amount * params['learner_population'] * 5
   
    # TRAINING
    print("Start training...\n")

    for ep in tqdm(range(1, train_episodes + 1), desc="EPISODES", colour='red', position=0, leave=False):
        env.reset()
        losses = []

        for tick in tqdm(range(1, params['episode_ticks'] + 1), desc="TICKS", colour='green', position=1, leave=False):
            for agent in env.agent_iter(max_iter=params['learner_population']):
                next_state, reward, _, _, _ = env.last(agent)
                next_state = torch.tensor(next_state, dtype=torch.float32, device=device)

                if ep == 1 and tick == 1:
                    #next_action = env.action_space(agent).sample()
                    idx = np.random.randint(0, actions_dim)
                    next_action = actions[idx]
                    next_action = torch.tensor([next_action], dtype=torch.long, device=device).unsqueeze(0)
                else:
                    state = old_s[agent]
                    action = old_a[agent]
                    #next_action, policy_nets[int(agent)] = select_action(env, agent, actions_dim, next_state, ep, policy_nets[int(agent)], device, epsilon_min, decay)
                    idx, policy_nets[int(agent)] = select_action(env, agent, actions_dim, next_state, ep, policy_nets[int(agent)], device, epsilon_min, decay)
                    next_action = actions[idx]
                    next_action = torch.tensor([next_action], dtype=torch.long, device=device).unsqueeze(0)
                    
                    #normalization is done considering the max reward a single agent can receive
                    reward = torch.tensor([reward], device=device) if not normalize \
                        else torch.tensor([reward / max_possible_reward], device=device)
                    
                    # Store the transition in memory
                    memory[int(agent)].push(state, action, next_state, reward)
                    
                    # Perform one step of the optimization (on the policy network)
                    policy_nets[int(agent)], target_nets[int(agent)], loss_single = optimize_model(Transition, memory[int(agent)], policy_nets[int(agent)], target_nets[int(agent)], gamma, batch_size, device)
                    if loss_single is not None:
                        # Optimize the model
                        optimizers[int(agent)].zero_grad()
                        loss_single.backward()
                        losses.append(torch.Tensor.clone(loss_single.detach()))
                        
                        # In-place gradient clipping
                        nn.utils.clip_grad_value_(policy_nets[int(agent)].parameters(), 100)
                        optimizers[int(agent)].step()
                        schedulers[int(agent)].step()
                    
                    # Soft update of the target network's weights
                    # θ′ ← τ θ + (1 −τ )θ′
                    if (int(agent) + tick * learner_population + ep * params['episode_ticks'] * learner_population) % update_net_every == 0:
                        target_net_state_dict = target_nets[int(agent)].state_dict()
                        policy_net_state_dict = policy_nets[int(agent)].state_dict()
                        for key in policy_net_state_dict:
                            target_net_state_dict[key] = policy_net_state_dict[key] * alpha + target_net_state_dict[key] * (1 - alpha)
                        target_nets[int(agent)].load_state_dict(target_net_state_dict)
                    
                epsilon = policy_nets[int(agent)].epsilon
                cur_lr = optimizers[int(agent)].param_groups[0]['lr']
                    
                env.step(next_action.item())
                old_s[agent] = next_state
                old_a[agent] = next_action

                actions_dict[str(ep)][str(idx)] += 1
                #actions_dict[str(ep)][str(next_action.item())] += 1
                #action_dict[str(ep)][str(agent)][str(next_action.item())] += 1
                reward_dict[str(ep)][str(agent)] += round(reward.item(), 2) if isinstance(reward, torch.Tensor) else round(reward, 2)
            
            if visualizer != None:
                visualizer.render(
                    env.patches,
                    env.learners,
                    env.turtles,
                    env.fov,
                    env.ph_fov
                )
        
        if ep % train_log_every == 0:
            print("EPISODE: {}\tepsilon: {:.5f}\tavg loss: {:.8f}\tlearning rate {:.10f}".format(ep, epsilon, sum(losses)/len(losses), cur_lr))
        #    avg_rew = round((sum(reward_dict[str(ep)].values()) / params["episode_ticks"]) / params["learner_population"], 2)
        #    avg_cluster = round(env.avg_cluster2(), 2)
        #    eps = round(epsilon, 4)
        #    value = [ep, tick * ep, avg_cluster, avg_rew]
        #    value.extend(list(actions_dict[str(ep)].values()))
        #    value.append(eps)
        #    logger.load_value(value)
            
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

    #logger.empty_table()
    env.close()
    if visualizer != None:
        visualizer.close()
    print("Training finished!\n")

    return policy_nets