import torch
from torch import nn
import torch.nn.functional as F
from collections import deque

import math
import random

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


def select_action(env, agent, state, steps_done, policy_net, device, epsilon_end, decay):
    sample = random.random()
    policy_net.epsilon = epsilon_end + (policy_net.epsilon - epsilon_end) * math.exp(-1. * steps_done * decay)
    
    if sample > policy_net.epsilon:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1), policy_net
    else:
        return torch.tensor([[env.action_space(agent).sample()]], device=device, dtype=torch.long), policy_net