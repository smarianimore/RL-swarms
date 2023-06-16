from slime_environments.environments.SlimeEnvMultiAgent import Slime

import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.utils import positional_encoding, read_params, setup
from utils.DQN import DQN, ReplayMemory

import argparse

import os
import math
import json
import random
import datetime
from collections import namedtuple
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


def select_action(env, agent, state, steps_done, policy_net, device, epsilon_end, decay):
    sample = random.random()
    eps_threshold = epsilon_end + (policy_net.epsilon - epsilon_end) * math.exp(-1. * steps_done * decay)
    
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1), policy_net
    else:
        return torch.tensor([[env.action_space(agent).sample()]], device=device, dtype=torch.long), policy_net


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
    
    
def train(env, 
          params, 
          l_params, 
          device, 
          policy_net, 
          target_net, 
          curdir,
          train_episodes,
          train_log_every,
          output_file):
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    batch_size = l_params["batch_size"]
    learning_rate = l_params["lr"]
    epsilon_end = l_params["epsilon_end"]
    alpha = l_params["alpha"]
    gamma = l_params["gamma"]
    decay = l_params["decay"]
    n_actions = len(l_params["actions"])
    population = params['population']
    learner_population = params['learner_population']
    
    optimizer = optim.AdamW(policy_net.parameters(), lr=learning_rate, amsgrad=True)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9999999)
    memory = {i: ReplayMemory(Transition, batch_size) for i in range(params['learner_population'])}
    
    old_s = {}
    old_a = {}
    cluster_dict = {}
    
    actions_dict = {str(ep): {str(ac): 0 for ac in range(n_actions)} for ep in range(1, train_episodes + 1)}  # DOC 0 = walk, 1 = lay_pheromone, 2 = follow_pheromone
    action_dict = {str(ep): {str(ag): {str(ac): 0 for ac in range(n_actions)} for ag in range(population, population + learner_population)} for ep in range(1, train_episodes + 1)}
    reward_dict = {str(ep): {str(ag): 0 for ag in range(population, population + learner_population)} for ep in range(1, train_episodes + 1)}
   
   
    for ep in range(1, train_episodes + 1):
        env.reset()
        losses = []
        
        # Initialize the environment and get it's state
        for tick in tqdm(range(1, params['episode_ticks'] + 1), desc=f"epsilon: {policy_net.epsilon}"):
            for agent in env.agent_iter(max_iter=params['learner_population']):
                next_state, reward, _, _  = env.last(agent)
                next_state = torch.tensor(next_state.observe(), dtype=torch.float32, device=device)
                
                new_pherormone = torch.tensor(env.get_neighborood_chemical(agent).reshape(-1,1), dtype=torch.float32).to(device).unsqueeze(0)
                pos_encoding = torch.tensor(positional_encoding(new_pherormone.numel(), 2), dtype=torch.float32).to(device).unsqueeze(0)
                new_pherormone = pos_encoding + new_pherormone
                
                next_state = torch.cat((torch.flatten(new_pherormone), next_state)).unsqueeze(0)
                
                if ep == 1 and tick == 1:
                    next_action = env.action_space(agent).sample()
                    next_action = torch.tensor([next_action], dtype=torch.long, device=device).unsqueeze(0)
                else:
                    state = old_s[agent]
                    action = old_a[agent]
                    next_action, policy_net = select_action(env, agent, next_state, ep, policy_net, device, epsilon_end, decay)
                    reward = torch.tensor([reward], device=device)
                    
                    # Store the transition in memory
                    memory[agent].push(state, action, next_state, reward)
                    
                    # Perform one step of the optimization (on the policy network)
                    policy_net, target_net, loss_single = optimize_model(Transition, memory[agent], policy_net, target_net, gamma, batch_size, device)
                    if loss_single is not None:
                        # Optimize the model
                        optimizer.zero_grad()
                        loss_single.backward()
                        losses.append(torch.Tensor.clone(loss_single.detach()))
                        memory[agent].memory.clear()
                        
                        # In-place gradient clipping
                        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
                        optimizer.step()
                        scheduler.step()
                    
                    # Soft update of the target network's weights
                    # θ′ ← τ θ + (1 −τ )θ′
                    target_net_state_dict = target_net.state_dict()
                    policy_net_state_dict = policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key] * alpha + target_net_state_dict[key] * (1 - alpha)
                    target_net.load_state_dict(target_net_state_dict)
                    
                env.step(next_action.item())
                old_s[agent] = next_state
                old_a[agent] = next_action
                
                policy_net.epsilon = epsilon_end + (policy_net.epsilon - epsilon_end) * math.exp(-1. * ep * decay)
                
                actions_dict[str(ep)][str(next_action.item())] += 1
                action_dict[str(ep)][str(agent)][str(next_action.item())] += 1
                reward_dict[str(ep)][str(agent)] += round(reward.item(), 2) if isinstance(reward, torch.Tensor) else round(reward, 2)                
                
            env.move()
            env._evaporate()
            env._diffuse()
            env.render()
            
            
        cluster_dict[str(ep)] = round(env.avg_cluster(), 2)
        if ep % train_log_every == 0:
            cur_lr = optimizer.param_groups[0]['lr']
            print("EPISODE: {}\tepsilon: {:.5f}\tavg loss: {:.3f}\tlearning rate {:.10f}".format(ep, policy_net.epsilon, sum(losses)/len(losses), cur_lr))
            
            with open(output_file, 'a') as f:
                f.write(f"{ep}, {params['episode_ticks'] * ep}, {cluster_dict[str(ep)]}, {actions_dict[str(ep)]['2']}, {actions_dict[str(ep)]['0']}, {actions_dict[str(ep)]['1']}, ")
                avg_rew = 0
                
                for l in range(params['population'], params['population'] + params['learner_population']):
                    avg_rew += (reward_dict[str(ep)][str(l)] / params['episode_ticks'])
                    f.write(f"{action_dict[str(ep)][str(l)]['2']}, {action_dict[str(ep)][str(l)]['0']}, {action_dict[str(ep)][str(l)]['1']}, ")
                
                avg_rew /= params['learner_population']
                f.write(f"{avg_rew}, {sum(losses)/len(losses)}, {cur_lr}\n")
                    
    #print(json.dumps(cluster_dict, indent=2))
    print("Training finished!\n")
    
    policy_model_name = "policy_"  + datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S") + ".pth"
    target_model_name = "target_"  + datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S") + ".pth"
    torch.save(policy_net.state_dict(), os.path.join(curdir, "models", policy_model_name))
    torch.save(target_net.state_dict(), os.path.join(curdir, "models", target_model_name))

    return policy_net, env


def test(env, params, l_params, policy_net, test_episodes, test_log_every, device):
    cluster_dict = {}
    print("[INFO] Start testing...")
    
    epsilon_end = l_params["epsilon_end"]
    policy_net.epsilon = epsilon_test = l_params["epsilon_test"]
    decay = l_params["decay"]
    
    for ep in range(1, test_episodes + 1):
        env.reset()
        for tick in tqdm(range(1, params['episode_ticks'] + 1), desc=f"epsilon: {policy_net.epsilon}"):
            for agent in env.agent_iter(max_iter=params['learner_population']):
                state, reward, _, _  = env.last(agent)
                state = torch.tensor(state.observe(), dtype=torch.float32, device=device).unsqueeze(0)
                action, policy_net = select_action(env, agent, state, ep, policy_net, device, epsilon_end, decay)
                env.step(action)
                
            env.move()
            env._evaporate()
            env._diffuse()
            env.render()
            
        if ep % test_log_every == 0:
            print(f"EPISODE: {ep}")
            print(f"\tepsilon: {policy_net.epsilon}")
            # print(f"\tepisode reward: {reward_episode}")
        cluster_dict[str(ep)] = round(env.avg_cluster(), 2)
        
    print(json.dumps(cluster_dict, indent=2))
    print("Testing finished!\n")


def main(args):
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    params, l_params = read_params(args.params_path, args.learning_params_path)
    curdir = os.path.dirname(os.path.abspath(__file__))
    output_file, alpha, gamma, epsilon, decay, train_episodes, train_log_every, test_episodes, test_log_every = setup(curdir, params, l_params)
    env = Slime(render_mode="human", **params)    
    
    if not os.path.isdir(os.path.join(curdir, "models")):
        os.makedirs(os.path.join(curdir, "models"))
    
    n_actions = len(l_params["actions"])
    n_observations = 100
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device selected: {device}")
    
    policy_net = DQN(n_observations, n_actions, epsilon).to(device)
    target_net = DQN(n_observations, n_actions, epsilon).to(device)
    
    if args.resume or args.test:
        if os.path.isfile(os.path.join(curdir, "models", args.policy_model_name)) and \
            os.path.isfile(os.path.join(curdir, "models", args.target_model_name)):
            policy_model_path = os.path.join(curdir, "models", args.policy_model_name)
            target_model_path = os.path.join(curdir, "models", args.target_model_name)
            policy_net.load_state_dict(torch.load(policy_model_path), strict=False)
            target_net.load_state_dict(torch.load(target_model_path), strict=False)
    else:
        target_net.load_state_dict(policy_net.state_dict())
    
    if args.train:
        policy_net, env = train(env, params, l_params, device, policy_net, target_net, curdir, train_episodes, train_log_every, output_file)
        
    if args.test:
        test(env, params, l_params, policy_net, test_episodes, test_log_every, device)

    env.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("params_path", type=str)
    parser.add_argument("learning_params_path", type=str)
    parser.add_argument("--policy-model-name", type=str, default="")
    parser.add_argument("--target-model-name", type=str, default="")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--random-seed", type=int, default=0)
    
    args = parser.parse_args()
    
    assert args.params_path != "" and os.path.isfile(args.params_path) and args.params_path.endswith(".json"), "[ERROR] params path is empty or is not a file or is not a json file"
    assert args.learning_params_path != "" and os.path.isfile(args.learning_params_path) and args.learning_params_path.endswith(".json"), "[ERROR] learning params path is empty or is not a file or is not a json file"
    
    main(args)