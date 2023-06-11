from slime_environments.environments.SlimeEnvMultiAgent import Slime

import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.utils import read_params, setup, state_to_int_map
from DQN import DQN, ReplayMemory

import argparse

import os
import math
import random
import datetime
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim

def select_action(env, agent, state, steps_done, l_params, policy_net, device):
    sample = random.random()
    eps_threshold = l_params["epsilon_end"] + (l_params["epsilon"] - l_params["epsilon_end"]) * math.exp(-1. * steps_done / l_params["decay"])
    steps_done += 1
    
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return steps_done, policy_net(state).max(1)[1].view(1, 1), policy_net, eps_threshold
    else:
        return steps_done, torch.tensor([[env.action_space(agent).sample()]], device=device, dtype=torch.long), policy_net, eps_threshold


def optimize_model(Transition, memory, policy_net, target_net, optimizer, gamma, batch_size, device):
    if len(memory) < batch_size:
        return
    
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

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    
    return policy_net, target_net, optimizer
    
    
def main(args):
    params, l_params = read_params(args.params_path, args.learning_params_path)
    env = Slime(render_mode="human", **params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device selected: {device}")

    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    output_file, alpha, gamma, epsilon, decay, train_episodes, train_log_every, test_episodes, test_log_every = setup(params, l_params)

    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor as mentioned in the previous section
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the ``AdamW`` optimizer
    batch_size = l_params["batch_size"]
    learning_rate = l_params["lr"]
    epsilon = l_params["epsilon"]

    # Get number of actions from gym action space
    n_actions = len(l_params["actions"])
    # Get the number of state observations
    n_observations = 2
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    
    curdir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isdir(os.path.join(curdir, "models")):
        os.makedirs(os.path.join(curdir, "models"))
        
    if os.path.isfile(os.path.join(curdir, "models", args.policy_model_name)) and \
        os.path.isfile(os.path.join(curdir, "models", args.target_model_name)):
        policy_model_path = os.path.join(curdir, "models", args.policy_model_name)
        target_model_path = os.path.join(curdir, "models", args.target_model_name)
        policy_net.load_state_dict(torch.load(policy_model_path), strict=False)
        target_net.load_state_dict(torch.load(target_model_path), strict=False)
    else:
        target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=learning_rate, amsgrad=True)
    
    memory = {i: ReplayMemory(Transition, 1000) for i in range(params['learner_population'])}
    old_s = {}
    cluster_dict = {}
    steps_done = 0    
    for ep in range(1, test_episodes + 1):
        env.reset()
        # Initialize the environment and get it's state
        for tick in range(1, params['episode_ticks'] + 1):
            for agent in env.agent_iter(max_iter=params['learner_population']):
                next_state, reward, _, _  = env.last(agent)
                next_state = torch.tensor(next_state.observe(), dtype=torch.float32, device=device).unsqueeze(0)
                
                if ep == 1 and tick == 1:
                    action = env.action_space(agent).sample()
                    action = torch.tensor([action], dtype=torch.long, device=device).unsqueeze(0)
                else:
                    state = old_s[agent]
                    steps_done, action, policy_net, epsilon = select_action(env, agent, next_state, steps_done, l_params, policy_net, device)
                    reward = torch.tensor([reward], device=device)
                    
                    # Store the transition in memory
                    memory[agent].push(state, action, next_state, reward)
                    
                    # Perform one step of the optimization (on the policy network)
                    policy_net, target_net, optimizer = optimize_model(Transition, memory[agent], policy_net, target_net, optimizer, gamma, batch_size, device)

                    # Soft update of the target network's weights
                    # θ′ ← τ θ + (1 −τ )θ′
                    target_net_state_dict = target_net.state_dict()
                    policy_net_state_dict = policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*alpha + target_net_state_dict[key]*(1-alpha)
                    target_net.load_state_dict(target_net_state_dict)
                    
                env.step(action.item())
                old_s[agent] = next_state
            env.move()
            env._evaporate()
            env._diffuse()
            env.render()
            
        cluster_dict[str(ep)] = round(env.avg_cluster(), 2)
        if ep % train_log_every == 0:
            print(f"EPISODE: {ep}")
            print(f"\tepsilon: {epsilon}")
                    
    #print(json.dumps(cluster_dict, indent=2))
    print("Training finished!\n")
    
    policy_model_name = "policy_"  + datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S") + ".pth"
    target_model_name = "target_"  + datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S") + ".pth"
    torch.save(policy_net.state_dict(), os.path.join(curdir, "models", policy_model_name))
    torch.save(target_net.state_dict(), os.path.join(curdir, "models", target_model_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("params_path", type=str)
    parser.add_argument("learning_params_path", type=str)
    parser.add_argument("--policy-model-name", type=str, default="")
    parser.add_argument("--target-model-name", type=str, default="")
    
    args = parser.parse_args()
    
    assert args.params_path != "" and os.path.isfile(args.params_path) and args.params_path.endswith(".json"), "[ERROR] params path is empty or is not a file or is not a json file"
    assert args.learning_params_path != "" and os.path.isfile(args.learning_params_path) and args.learning_params_path.endswith(".json"), "[ERROR] learning params path is empty or is not a file or is not a json file"
    
    main(args)