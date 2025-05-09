import json
import random
import sys
from typing import Optional
from pprint import pprint
import time
import cProfile

import gymnasium as gym
from gymnasium.spaces import Discrete, MultiBinary, Box

import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import ObsType
from pettingzoo.test import api_test

from scipy.ndimage import gaussian_filter

class Slime(AECEnv):
    def observe(self, agent: str) -> ObsType:
        return np.array(self.observations[agent])

    def observation_space(self, agent):
        return self._observation_spaces[agent]
    
    def action_space(self, agent):
        return self._action_spaces[agent]
      
    def observations_n(self, same_obs=True):
        if same_obs:
            if isinstance(self.observation_space('0'), MultiBinary):
                return self.observation_space('0').n
            elif isinstance(self.observation_space('0'), Box):
                return self.observation_space('0').shape[0]

    def actions_n(self, same_actions=True):
        if same_actions:
            return self.action_space('0').n.item()

    metadata = {"render_modes": ["human", "server"]}

    def __init__(self, seed, render_mode: Optional[str] = None, **kwargs):
        """
        :param cluster_learners     Controls the number of clustering agents (red turtles). 
        :param scatter_learners     Controls the number of scattering agents (blue turtles).
        :param actions              Controls the actions performed by the agents.
        :param sniff_threshold      Controls how sensitive slimes are to pheromone (higher values make slimes less
                                    sensitive to pheromone)—unclear effect on learning, could be negligible.
        :param sniff_patches        Controls the number of 1-hop neighboring patches in which the agent can smell the 
                                    pheromone.
        :param wiggle_patches       Controls the number of 1-hop neighboring patches the agent can move randomly through.
        :param diffuse_area         Controls the standard deviation value (std) of the Gaussian function used to spread 
                                    the pheromone in the environment.
        :param diffure_radius       Controls the radius of the Gaussian function used to spread the pheromone in the 
                                    environment.
        :param lay_area             Controls the radius of the square area sorrounding the turtle where pheromone is laid.
        :param lay_amount           Controls how much pheromone is laid.
        :param evaporation          Controls how much pheromone evaporates at each step.
        :param cluster_threshold    Controls the minimum number of slimes needed to consider an aggregate within
                                    cluster-radius a cluster (the higher the more difficult to consider an aggregate a
                                    cluster)—the higher the more difficult to obtain a positive reward for being within
                                    a cluster for learning slimes.
        :param cluster_radius       Controls the range considered by slimes to count other slimes within a cluster (the
                                    higher the easier to form clusters, as turtles far apart are still counted together)
                                    —the higher the easier it is to obtain a positive reward for being within a cluster
                                    for learning slimes.
        :param normalize_rewards    If True the rewards are normalized (max value = 1.0).
        :param cluster_rew:         Clustering agent base reward for being in a cluster.
        :param cluster_penalty:     Clustering agent base penalty for not being in a cluster.
        :param scatter_rew:         Scattering agent base reward for not being in a cluster.
        :param scatter_penalty:     Scattering agent base penalty for being in a cluster.
        :param episode_ticks:       Number of ticks for episode termination.
        :param W:                   Window width in # patches.
        :param H:                   Window height in # patches.
        :param PATCH_SIZE:          Patch size in pixels.
        :param TURTLE_SIZE:         Turtle size in pixels.
        """

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        np.random.seed(seed)
        random.seed(seed)
        
        self.population = kwargs['population']
        self.cluster_learners = kwargs['cluster_learners'] 
        self.scatter_learners = kwargs['scatter_learners'] 
        self.sniff_threshold = kwargs['sniff_threshold']
        self.diffuse_area = kwargs['diffuse_area']
        self.diffuse_radius = kwargs['diffuse_radius']
        self.smell_area = kwargs['smell_area']
        self.lay_area = kwargs['lay_area']
        self.lay_amount = kwargs['lay_amount']
        self.evaporation = kwargs['evaporation']
        self.diffuse_mode = kwargs['diffuse_mode']
        self.follow_mode = kwargs['follow_mode']
        self.cluster_threshold = kwargs['cluster_threshold']
        self.cluster_radius = kwargs['cluster_radius']
        self.normalize_rewards = kwargs['normalize_rewards']
        self.episode_ticks = kwargs['episode_ticks']
    
        self.cluster_reward = kwargs['cluster_rew']
        self.cluster_penalty = kwargs['cluster_penalty']
        self.scatter_reward = kwargs['scatter_rew']
        self.scatter_penalty = kwargs['scatter_penalty']

        self.W = kwargs['W']
        self.H = kwargs['H']
        self.patch_size = kwargs['PATCH_SIZE']
        self.turtle_size = kwargs['TURTLE_SIZE']

        self.N_DIRS = 8
        self.sniff_patches = kwargs['sniff_patches']
        self.wiggle_patches = kwargs['wiggle_patches'] 
        assert (
            self.sniff_patches in (1, 3, 5, 7, 8)
        ), "Error! sniff_patches admitted values are: 1, 3, 5, 7, 8."
        assert (
            self.wiggle_patches in (1, 3, 5, 7, 8)
        ), "Error! wiggle_patches admitted values are: 1, 3, 5, 7, 8."
        # Used to calculate the agent's directions.
        # It's a personal convention.

        self.coords = []
        self.offset = self.patch_size // 2
        self.W_pixels = self.W * self.patch_size
        self.H_pixels = self.H * self.patch_size
        for x in range(self.offset, (self.W_pixels - self.offset) + 1, self.patch_size):
            for y in range(self.offset, (self.H_pixels - self.offset) + 1, self.patch_size):
                self.coords.append((x, y))  # "centre" of the patch or turtle (also ID of the patch)

        pop_tot = self.population + self.cluster_learners + self.scatter_learners
        self.possible_agents = [str(i) for i in range(self.population, pop_tot)]  # DOC learning agents IDs
        self._agent_selector = agent_selector(self.possible_agents)
        self.agent = self._agent_selector.reset()

        n_coords = len(self.coords)
        # create learners turtle
        self.learners = {
            i: {
                "pos": self.coords[np.random.randint(n_coords)],
                "dir": np.random.randint(self.N_DIRS), 
                "mode": 'c' if i < self.cluster_learners else 's'
            } for i in range(self.cluster_learners + self.scatter_learners)
        }

        # patches-own [chemical] - amount of pheromone in each patch
        self.patches = {self.coords[i]: {"id": i,
                                         'chemical': 0.0,
                                         'turtles': []} for i in range(n_coords)}
        for l in self.learners:
            self.patches[self.learners[l]['pos']]['turtles'].append(l)  # DOC id of learner turtles

        # pre-compute relevant structures to speed-up computation during rendering steps
        # DOC {(x,y): [(x,y), ..., (x,y)]} pre-computed smell area for each patch, including itself
        self.smell_patches = self._find_neighbours(self.smell_area)
        # DOC {(x,y): [(x,y), ..., (x,y)]} pre-computed lay area for each patch, including itself
        self.lay_patches = self._find_neighbours(self.lay_area)
        # DOC {(x,y): [(x,y), ..., (x,y)]} pre-computed cluster-check for each patch, including itself
        self.cluster_patches = self._find_neighbours(self.cluster_radius)

        # Agent's field of view
        self.fov = self._field_of_view(self.wiggle_patches)
        # Agent's pheromone field of view
        self.ph_fov = self._field_of_view(self.sniff_patches)

        self.actions = kwargs['actions']
        self._action_spaces = {
            a: Discrete(len(self.actions))
            for a in self.possible_agents
        }  # DOC 0 = walk, 1 = lay_pheromone, 2 = follow_pheromone
        
        assert kwargs['obs_type'] in ("paper", "variation1", "variation2")
        self.obs_type = kwargs['obs_type']
        # DOC obervation is an array of 8 real elements.
        # This array indicates the pheromone values in the 8 patches around the agent.
        if self.obs_type == "paper":
            self._observation_spaces = {
                a: Box(low=0.0, high=np.inf, shape=(self.sniff_patches,), dtype=np.float32)
                for a in self.possible_agents
            }
        elif self.obs_type == "variation1":
            self._observation_spaces = {
                a: Box(low=0.0, high=np.inf, shape=(self.sniff_patches + 1,), dtype=np.float32)
                for a in self.possible_agents
            }
        elif self.obs_type == "variation2":
            self._observation_spaces = {
                a: MultiBinary(2)
                for a in self.possible_agents
            }  # DOC [0] = whether the turtle is in a cluster [1] = whether there is chemical in turtle patch

        #self.REWARD_MAX = self.reward + (((self.cluster_learners - 1) / self.cluster_threshold) * (self.reward ** 2))
        self.REWARD_MAX = self.cluster_reward + (((self.cluster_learners - 1) / self.cluster_threshold) * (self.cluster_reward ** 2))

        #Different from AECEnv attribute self.rewards - only keeps last step rewards
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(self.population, pop_tot)))
        )

    def _field_of_view(self, n_patches):
        """
        For every possible agent's direction, pre-compute the group of patched, based on 'n_patches' value.
        """

        # Pre-compute every possible agent's direction
        movements = np.array([
            (0, -self.patch_size),                  # dir 0
            (self.patch_size, -self.patch_size),    # dir 1
            (self.patch_size, 0),                   # dir 2
            (self.patch_size, self.patch_size),     # dir 3
            (0, self.patch_size),                   # dir 4
            (-self.patch_size, self.patch_size),    # dir 5
            (-self.patch_size, 0),                  # dir 6
            (-self.patch_size, -self.patch_size),   # dir 7
        ])
        fov = {}
        
        if n_patches < self.N_DIRS:
            central = n_patches // 2
            sliding_window = []
            
            for i in range(self.N_DIRS):
                tmp = []
                for j in range(n_patches):
                    tmp.append((i + j) % self.N_DIRS)
                sliding_window.append(tmp)
            sliding_window = sorted(sliding_window, key=lambda x: x[central])
            
            for c in self.coords:
                tmp_fov = movements + c
                tmp_fov[:, 0] %= self.W_pixels 
                tmp_fov[:, 1] %= self.H_pixels 
                fov[c] = tmp_fov[sliding_window, :]
        else:
            for c in self.coords:
                tmp_fov = movements + c
                tmp_fov[:, 0] %= self.W_pixels 
                tmp_fov[:, 1] %= self.H_pixels 
                fov[c] = tmp_fov

        return fov

    def _find_neighbours(self, area: int):
        """
        For each patch, find neighbouring patches within square radius 'area'
        """

        neighbours = {}
        
        for p in self.patches:
            neighbours[p] = []
            for x in range(p[0], p[0] + (area * self.patch_size) + 1, self.patch_size):
                for y in range(p[1], p[1] + (area * self.patch_size) + 1, self.patch_size):
                    x, y = self._wrap(x, y)
                    neighbours[p].append((x, y))
            for x in range(p[0], p[0] - (area * self.patch_size) - 1, -self.patch_size):
                for y in range(p[1], p[1] - (area * self.patch_size) - 1, -self.patch_size):
                    x, y = self._wrap(x, y)
                    neighbours[p].append((x, y))
            for x in range(p[0], p[0] + (area * self.patch_size) + 1, self.patch_size):
                for y in range(p[1], p[1] - (area * self.patch_size) - 1, -self.patch_size):
                    x, y = self._wrap(x, y)
                    neighbours[p].append((x, y))
            for x in range(p[0], p[0] - (area * self.patch_size) - 1, -self.patch_size):
                for y in range(p[1], p[1] + (area * self.patch_size) + 1, self.patch_size):
                    x, y = self._wrap(x, y)
                    neighbours[p].append((x, y))
            neighbours[p] = list(set(neighbours[p]))

        return neighbours

    def _wrap(self, x: int, y: int):
        """
        Wrap x,y coordinates around the torus

        :param x: the x coordinate to wrap
        :param y: the y coordinate to wrap
        :return: the wrapped x, y
        """
        return x % self.W_pixels, y % self.H_pixels

    # learners act
    def step(self, action: int):
        if(self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
            self._was_dead_step(action)
            return
        
        self.agent = self.agent_name_mapping[self.agent_selection]  # ID of agent

        self.observations[str(self.agent)], self.cluster_ticks, self.rewards_cust = self.process_agent(
            self.cluster_ticks,
            self.rewards_cust,
        )
        
        if action == 0:     # Walk
            self.do_action0()
        elif action == 1:   # Lay pheromone
            self.do_action1()
        elif action == 2:   # Follow pheromone
            self.do_action2()
        elif action == 3:   # Don't follow pheromone
            self.do_action3()
        elif action == 4:   # Walk and Lay pheromone
            self.do_action0()
            self.do_action1()
        elif action == 5:   # Follow pheromone and Lay pheromone 
            self.do_action2()
            self.do_action1()
        else:
            raise ValueError("Action out of range!")

        if self._agent_selector.is_last():
            for ag in self.agents:
                self.rewards[ag] = self.rewards_cust[self.agent_name_mapping[ag]][-1]

            self.patches = self._diffuse_and_evaporate(self.patches)
        else:
            self._clear_rewards()
            
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[str(self.agent)] = 0
        self._accumulate_rewards()
        
    def _get_new_positions(self, possible_patches, agent):
        pos = agent["pos"]
        direction = agent["dir"]
        if len(possible_patches[pos].shape) > 2:
            return possible_patches[pos][direction], direction
        else:
            return possible_patches[pos], direction
    
    def _get_new_direction(self, n_patches, old_dir, idx_dir):
        start = (old_dir - (n_patches // 2)) % self.N_DIRS 
        new_dirs = np.array([(i + start) % self.N_DIRS for i in range(n_patches)])
        return new_dirs[idx_dir]

    def process_agent(self, cluster_ticks, rewards_cust):
        """
        In this methods we compute the agent's reward and it's observation.
        """
        cluster = self._compute_cluster(self.agent)

        if self.learners[self.agent]["mode"] == 'c':
            cluster_ticks, rewards_cust, cur_reward = self.reward_cluster_and_time_punish_time(
                cluster_ticks,
                rewards_cust,
                cluster
            )
        elif self.learners[self.agent]["mode"] == 's':
            cluster_ticks, rewards_cust, cur_reward = self.reward_scatter_and_time_punish_time(
                cluster_ticks,
                rewards_cust,
                cluster
            )
        
        if self.obs_type == "paper":
            observations = self._get_obs(self.learners[self.agent])
        elif self.obs_type == "variation1":
            observations = self._get_obs2(self.learners[self.agent])
        elif self.obs_type == "variation2":
            chemical = self._check_chemical(self.agent)
            observations = np.array([cluster >= self.cluster_threshold, chemical])

        return observations, cluster_ticks, rewards_cust

    def _get_obs(self, agent):
        f, _ = self._get_new_positions(self.ph_fov, agent)
        obs = np.array([self.patches[tuple(i)]["chemical"] for i in f])
        return obs

    def _get_obs2(self, agent):
        f, _ = self._get_new_positions(self.ph_fov, agent)
        obs = [self.patches[tuple(i)]["chemical"] for i in f]
        obs.append(self.patches[agent["pos"]]["chemical"])
        return np.array(obs)

    def do_action0(self):
        self.patches, self.learners[self.agent] = self.walk(self.patches, self.learners[self.agent])

    def do_action1(self):
        self.patches = self.lay_pheromone(self.patches, self.learners[self.agent]['pos'])

    def do_action2(self):
        if self.obs_type == "paper":
            max_pheromone, max_coords, max_ph_dir = self._find_max_pheromone(
                self.learners[self.agent],
                self.observations[str(self.agent)]       
            )
            if max_pheromone >= self.sniff_threshold:
                self.patches, self.learners[self.agent] = self.follow_pheromone(
                    self.patches,
                    max_coords,
                    max_ph_dir,
                    self.learners[self.agent]
                )
            else:
                self.do_action0()
        elif self.obs_type == "variation1":
            max_pheromone, max_coords, max_ph_dir = self._find_max_pheromone2(
                self.learners[self.agent],
                self.observations[str(self.agent)]       
            )
            if max_pheromone >= self.sniff_threshold:
                if max_coords != self.learners[self.agent]["pos"]:
                    self.patches, self.learners[self.agent] = self.follow_pheromone(
                        self.patches,
                        max_coords,
                        max_ph_dir,
                        self.learners[self.agent]
                    )
            else:
                self.do_action0()

    def do_action3(self):
        if np.any(self.observations[str(self.agent)] >= self.sniff_threshold):
            ph_pos, ph_dir = self._find_non_max_pheromone(self.learners[self.agent], self.observations[str(self.agent)])
            self.patches, self.learners[self.agent] = self.avoid_pheromone(
                self.patches,
                ph_pos,
                ph_dir,
                self.learners[self.agent]
            )
        else:
            self.do_action0()

    def convert_observation(self, obs):
        if self.obs_type == "paper":
            if np.unique(obs).shape[0] == 1:
                obs_id = np.random.randint(self.sniff_patches)
            else:
                obs_id = obs.argmax().item()
        elif self.obs_type == "variation1":
            if np.unique(obs).shape[0] == 1:
                obs_id = np.random.randint(self.sniff_patches + 1)
            else:
                obs_id = obs.argmax().item()
        elif self.obs_type == "variation2":
            obs_id = int(f"{obs[0].astype(np.uint8)}{obs[1].astype(np.uint8)}", 2)
        return obs_id

    def lay_pheromone(self, patches, pos):
        """
        Lay 'amount' pheromone in square 'area' centred in 'pos'
        """
        for p in self.lay_patches[pos]:
            patches[p]['chemical'] += self.lay_amount
        
        return patches

    def _diffuse_and_evaporate(self, patches):
        """
        This method combine the _diffuse2 and _evaporate methods in one function.
        It is the method currently used.
        """
        # Diffusion
        grid = np.array([patches[p]["chemical"] for p in patches.keys()]).reshape((self.W, self.H))
        if self.diffuse_radius == 0:
            grid = gaussian_filter(grid, sigma=self.diffuse_area, mode="wrap")
        else:
            grid = gaussian_filter(grid, sigma=self.diffuse_area, radius=self.diffuse_radius, mode="wrap")
        grid = grid.flatten()
        # Evaporation
        grid *= self.evaporation
        # Write values
        for p, g in zip(patches, grid):
            patches[p]['chemical'] = g
        
        return patches

    def walk(self, patches, turtle):
        f, direction = self._get_new_positions(self.fov, turtle)
        idx_dir = np.random.randint(f.shape[0])
        patches[turtle['pos']]['turtles'].remove(self.agent)
        turtle["pos"] = tuple(f[idx_dir])
        patches[turtle['pos']]['turtles'].append(self.agent)
        if self.wiggle_patches < self.N_DIRS:
            turtle["dir"] = self._get_new_direction(self.wiggle_patches, direction, idx_dir)
        else:
            turtle["dir"] = idx_dir
        return patches, turtle

    def avoid_pheromone(self, patches, ph_coords, ph_dir, turtle):
        patches[turtle['pos']]['turtles'].remove(self.agent)
        turtle["pos"] = ph_coords
        patches[turtle['pos']]['turtles'].append(self.agent)
        turtle["dir"] = ph_dir
        return patches, turtle

    def follow_pheromone(self, patches, ph_coords, ph_dir, turtle):
        patches[turtle['pos']]['turtles'].remove(self.agent)
        turtle["pos"] = ph_coords
        patches[turtle['pos']]['turtles'].append(self.agent)
        turtle["dir"] = ph_dir
        return patches, turtle

    def _find_max_pheromone(self, agent, obs):
        # Det = follow greatest pheromone
        f, direction = self._get_new_positions(self.ph_fov, agent)
        if self.follow_mode == "prob": 
            total = obs.sum()
            if total == 0.0:
                probs = np.ones_like(obs) / obs.shape[0]
            else:
                probs = obs / obs.sum()
            idx = np.random.choice(np.arange(obs.shape[0]), p=probs)
        else:
            idx = obs.argmax()
        ph_val = obs[idx]
        ph_pos = tuple(f[idx])
        if self.sniff_patches < self.N_DIRS:
            ph_dir = self._get_new_direction(self.sniff_patches, direction, idx)
        else:
            ph_dir = idx
        return ph_val, ph_pos, ph_dir

    def _find_max_pheromone2(self, agent, obs):
        # Det = follow greatest pheromone
        if self.follow_mode == "prob": 
            total = obs.sum()
            if total == 0.0:
                probs = np.ones_like(obs) / obs.shape[0]
            else:
                probs = obs / obs.sum()
            idx = np.random.choice(np.arange(obs.shape[0]), p=probs)
        else:
            idx = obs.argmax()
        ph_val = obs[idx]
        if idx == (obs.shape[0] - 1):
            ph_pos = agent["pos"] 
            ph_dir = agent["dir"]
        else:
            f, direction = self._get_new_positions(self.ph_fov, agent)
            ph_pos = tuple(f[idx])
            if self.sniff_patches < self.N_DIRS:
                ph_dir = self._get_new_direction(self.sniff_patches, direction, idx)
            else:
                ph_dir = idx
        return ph_val, ph_pos, ph_dir

    def _find_non_max_pheromone(self, agent, obs):
        f, direction = self._get_new_positions(self.ph_fov, agent)
        ids = np.where(obs < self.sniff_threshold)[0]
        
        #if ids.shape[0] == obs.shape[0]:
        if ids.shape[0] == 0:
            idx = obs.argmin()
        else:
            idx = np.random.choice(ids)

        #ph_val = obs[idx]
        ph_pos = tuple(f[idx])
        if self.sniff_patches < self.N_DIRS:
            ph_dir = self._get_new_direction(self.sniff_patches, direction, idx)
        else:
            ph_dir = idx
        return ph_pos, ph_dir

    def _compute_cluster(self, current_agent):
        """
        Checks whether the learner turtle is within a cluster, given 'cluster_radius' and 'cluster_threshold'
        """
        #cluster = 0
        cluster = -1
        for p in self.cluster_patches[self.learners[current_agent]['pos']]:
            cluster += len(self.patches[p]['turtles'])
        
        return cluster

    def _compute_avg_cluster(self, clusters):
        cluster_sum = 0
        for cluster in clusters:
            cluster_sum += len(cluster)

        return cluster_sum / len(clusters)

    def _get_double_agent_clusters(self, clusters):
        only_cluster = []
        only_scatter = []
        mixed_cluster = []
        mixed_scatter = []
        
        for cluster in clusters:
            tmp_cluster = []
            tmp_scatter = []
            counter_cluster = True 
            counter_scatter = True 
            
            for c in cluster:
                if self.learners[c]["mode"] == 'c':
                    tmp_cluster.append(c)
                    
                    if counter_cluster:
                        mixed_cluster.append(cluster)
                        counter_cluster = False
                elif self.learners[c]["mode"] == 's':
                    tmp_scatter.append(c)
                    
                    if counter_scatter:
                        mixed_scatter.append(cluster)
                        counter_scatter = False
            if len(tmp_cluster) > 0:
                only_cluster.append(tmp_cluster)
            
            if len(tmp_scatter) > 0:
                only_scatter.append(tmp_scatter)
        
        return only_cluster, mixed_cluster, only_scatter, mixed_scatter

    def avg_cluster(self):
        """
        Same compuation as avg_cluster.
        Use THIS for calculating the average, avg_cluster has a bug!
        """
        cluster_sizes = []  # registra la dim. dei cluster
        for l in self.learners:
            cluster = []  # tiene conto di quali turtle sono in quel cluster
            for p in self.cluster_patches[self.learners[l]['pos']]:
                for t in self.patches[p]['turtles']:
                    cluster.append(t)
            #cluster.sort()
            if cluster not in cluster_sizes:
                cluster_sizes.append(cluster)
        
        cs = cluster_sizes.copy()
        for i in range(len(cluster_sizes)):
            for j in range(i + 1, len(cluster_sizes)):
                set1 = set(cluster_sizes[j])
                set2 = set(cluster_sizes[i])
                if set1.issubset(set2) and cluster_sizes[j] in cs:
                    cs.remove(cluster_sizes[j])
                elif set2.issubset(set1) and cluster_sizes[i] in cs:
                    cs.remove(cluster_sizes[i])
        
        # calcolo avg_cluster_size
        if self.cluster_learners == 0 or self.scatter_learners == 0:
            avg_cluster_size = self._compute_avg_cluster(cs)
            
            return avg_cluster_size
        else:
            (
                only_cluster,
                mixed_cluster,
                only_scatter,
                mixed_scatter
            ) = self._get_double_agent_clusters(cs)
            avg_only_cluster = self._compute_avg_cluster(only_cluster)
            avg_mixed_cluster = self._compute_avg_cluster(mixed_cluster)
            avg_only_scatter = self._compute_avg_cluster(only_scatter)
            avg_mixed_scatter = self._compute_avg_cluster(mixed_scatter)

            return avg_only_cluster, avg_mixed_cluster, avg_only_scatter, avg_mixed_scatter

    def _check_chemical(self, current_agent):
        """
        Checks whether there is pheromone on the patch where the learner turtle is
        """
        return self.patches[self.learners[current_agent]['pos']][
                   'chemical'] > self.sniff_threshold

    def reward_cluster_and_time_punish_time(self, cluster_ticks, rewards_cust, cluster):
        """
        The clustering reward used in the article.
        """
        if cluster >= self.cluster_threshold:
            cluster_ticks[self.agent] += 1

        cur_reward = (cluster_ticks[self.agent] / self.episode_ticks) * self.cluster_reward + \
                     (cluster / self.cluster_threshold) * (self.cluster_reward ** 2) + \
                     (((self.episode_ticks - cluster_ticks[self.agent]) / self.episode_ticks) * self.cluster_penalty)
        if self.normalize_rewards:
            cur_reward = round(cur_reward / self.REWARD_MAX, 4)

        rewards_cust[self.agent].append(cur_reward)
        return cluster_ticks, rewards_cust, cur_reward
    
    def reward_scatter_and_time_punish_time(self, cluster_ticks, rewards_cust, cluster):
        """
        The scattering reward used in the article.
        """
        if cluster >= self.cluster_threshold:
            cluster_ticks[self.agent] += 1

        cur_reward = (cluster_ticks[self.agent] / self.episode_ticks) * self.scatter_penalty - \
                     (cluster / self.cluster_threshold) * (self.scatter_penalty ** 2) + \
                     (((self.episode_ticks - cluster_ticks[self.agent]) / self.episode_ticks) * self.scatter_reward)
        if self.normalize_rewards:
            cur_reward = round(cur_reward / self.scatter_reward, 4)

        rewards_cust[self.agent].append(cur_reward)
        return cluster_ticks, rewards_cust, cur_reward

    def reset(self, seed=None, return_info=True, options=None):
        """
        Reset env.
        """
        # empty stuff
        pop_tot = self.population + self.cluster_learners + self.scatter_learners
        self.rewards_cust = {i: [] for i in range(self.population, pop_tot)}
        self.cluster_ticks = {i: 0 for i in range(self.population, pop_tot)}
        
        #Initialize attributes for PettingZoo Env
        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        
        # re-position learner turtle
        for l in self.learners:
            self.patches[self.learners[l]['pos']]['turtles'].remove(l)
            self.learners[l]['pos'] = self.coords[np.random.randint(len(self.coords))]
            self.patches[self.learners[l]['pos']]['turtles'].append(l)  # DOC id of learner turtle
        # patches-own [chemical] - amount of pheromone in the patch
        for p in self.patches:
            self.patches[p]['chemical'] = 0.0

        if self.obs_type == "paper":
            self.observations = {
                a: np.zeros(self.sniff_patches, dtype=np.float32)
                for a in self.agents
            }
        elif self.obs_type == "variation1":
            self.observations = {
                a: np.zeros(self.sniff_patches + 1, dtype=np.float32)
                for a in self.agents
            }
        elif self.obs_type == "variation2":
            self.observations = {a: np.full((2, ), False) for a in self.agents}
        
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

import pygame

BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
RED = (190, 0, 0)
GREEN = (0, 190, 0)
YELLOW = (250, 250, 0)

class SlimeVisualizer:
    def __init__(
        self,
        W_pixels,
        H_pixels,
        **kwargs
    ):
        self.fps = kwargs['FPS']
        self.shade_strength = kwargs['SHADE_STRENGTH']
        self.show_chem_text = kwargs['SHOW_CHEM_TEXT']
        self.cluster_font_size = kwargs['CLUSTER_FONT_SIZE']
        self.chemical_font_size = kwargs['CHEMICAL_FONT_SIZE']
        self.sniff_threshold = kwargs['sniff_threshold']
        #self.sniff_threshold = 0.0 
        self.patch_size = kwargs['PATCH_SIZE']
        self.turtle_size = kwargs['TURTLE_SIZE']

        self.W_pixels = W_pixels
        self.H_pixels = H_pixels
        self.offset = self.patch_size // 2
        self.screen = pygame.display.set_mode((self.W_pixels, self.H_pixels))
        self.clock = pygame.time.Clock()
        pygame.font.init()
        self.cluster_font = pygame.font.SysFont("arial", self.cluster_font_size)
        self.chemical_font = pygame.font.SysFont("arial", self.chemical_font_size)
        self.first_gui = True

        self.show_dirs_view = kwargs["show_dirs_view"]
        if self.show_dirs_view:
            self.N_DIRS = 8
            self.wiggle_patches = kwargs["wiggle_patches"]
            self.dirs = self._get_dirs()
        self.show_ph_view = kwargs["show_ph_view"]

    def _get_dirs(self):
        central = self.wiggle_patches // 2
        sliding_window = []
        
        for i in range(self.N_DIRS):
            tmp = []
            for j in range(self.wiggle_patches):
                tmp.append((i + j) % self.N_DIRS)
            sliding_window.append(tmp)
        
        sliding_window = sorted(sliding_window, key=lambda x: x[central])
        return np.array(sliding_window)

    def render(
        self,
        patches,
        learners,
        fov,
        ph_fov
    ):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # window closed -> program quits
                pygame.quit()

        if self.first_gui:
            self.first_gui = False
            pygame.init()
            pygame.display.set_caption("SLIME")

        self.screen.fill(BLACK)
        # draw patches
        for p in patches:
            chem = round(patches[p]['chemical']) * self.shade_strength
            pygame.draw.rect(
                self.screen,
                (0, chem if chem <= 255 else 255, 0),
                pygame.Rect(
                    p[0] - self.offset,
                    p[1] - self.offset,
                    self.patch_size,
                    self.patch_size
                )
            )
            if self.show_chem_text and (not sys.gettrace() is None or
                                        patches[p]['chemical'] >= self.sniff_threshold):  # if debugging show text everywhere, even 0
                text = self.chemical_font.render(str(round(patches[p]['chemical'], 1)), True, GREEN)
                self.screen.blit(text, text.get_rect(center=p))

        # draw learners
        for learner in learners.values():
            pygame.draw.circle(
                self.screen,
                RED if learner["mode"] == 'c' else BLUE,
                (learner['pos'][0], learner['pos'][1]),
                self.turtle_size // 2
            )

            if self.show_dirs_view:
                if len(fov[learner["pos"]].shape) > 2:
                    view = fov[learner["pos"]][learner["dir"]]
                    dirs = self.dirs[learner["dir"]]
                else:
                    view = fov[learner["pos"]]
                    dirs = self.dirs[4]
                
                for f, d in zip(view, dirs):
                    pygame.draw.rect(
                        self.screen,
                        YELLOW,
                        pygame.Rect(
                            f[0] - self.offset,
                            f[1] - self.offset,
                            self.patch_size,
                            self.patch_size
                        )
                    )
                    text = self.cluster_font.render(str(d), True, BLACK)
                    self.screen.blit(text, text.get_rect(center=f))

            if self.show_ph_view:
                if len(ph_fov[learner["pos"]].shape) > 2:
                    ph = ph_fov[learner["pos"]][learner["dir"]]
                else:
                    ph = ph_fov[learner["pos"]]
                
                for f in ph:
                    pygame.draw.rect(
                        self.screen,
                        WHITE,
                        pygame.Rect(
                            f[0] - self.offset,
                            f[1] - self.offset,
                            self.patch_size,
                            self.patch_size
                        )
                    )
                    if patches[learner["pos"]]['chemical'] >= self.sniff_threshold:
                        text = self.chemical_font.render(
                            str(round(patches[tuple(f)]['chemical'], 1)),
                            True,
                            BLACK
                        )
                        self.screen.blit(text, text.get_rect(center=f))

        for p in patches:
            if len(patches[p]['turtles']) > 1:
                text = self.cluster_font.render(str(len(patches[p]['turtles'])), True,
                                                RED if -1 in patches[p]['turtles'] else WHITE)
                self.screen.blit(text, text.get_rect(center=p))

        self.clock.tick(self.fps)
        pygame.display.flip()

        return pygame.surfarray.array3d(self.screen)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

def main():
    params = {
        "population": 0,
        #"learner_population": 50,
        "cluster_learners": 5,
        "scatter_learners": 5,
        "actions": [
            "move-toward-chemical",
            "random-walk",
            "drop-chemical",
            "move-away-chemical"
            #"move-and-drop",
            #"walk-and-drop",
        ],
        "sniff_threshold": 0.9,
        "sniff_patches": 8, 
        "diffuse_area": 0.5,
        "diffuse_radius": 0,
        "diffuse_mode": "gaussian",
        #"diffuse_mode": "cascade",
        "follow_mode": "det",
        #"follow_mode": "prob",
        "wiggle_patches": 8,
        "smell_area": 1,
        "lay_area": 1,
        "lay_amount": 3,
        "evaporation": 0.90,
        "cluster_threshold": 15,
        "cluster_radius": 3,
        #"obs_type": "variation1",
        "obs_type": "paper",
        "normalize_rewards": False,
        "cluster_rew": 10,
        "cluster_penalty": 0,
        "scatter_rew": 0,
        "scatter_penalty": -10,
        "episode_ticks": 500,
        #"W": 16,
        "W": 19,
        #"H": 16,
        "H": 19,
        "PATCH_SIZE": 20,
        #"PATCH_SIZE": 4,
        "TURTLE_SIZE": 16,
        #"TURTLE_SIZE": 3,
    }

    params_visualizer = {
      "FPS": 15,
      #"FPS": 3,
      "SHADE_STRENGTH": 10,
      "SHOW_CHEM_TEXT": False,
      "CLUSTER_FONT_SIZE": 12,
      "CHEMICAL_FONT_SIZE": 8,
      "gui": True,
      "sniff_threshold": 0.9,
      "PATCH_SIZE": 20,
      #"PATCH_SIZE": 4,
      "TURTLE_SIZE": 16,
      #"TURTLE_SIZE": 3,
      "show_dirs_view": False,
      "wiggle_patches": 3,
      "show_ph_view": False
    }

    from tqdm import tqdm

    EPISODES = 5
    LOG_EVERY = 1
    SEED = 0
    np.random.seed(SEED)
    env = Slime(SEED, **params)
    env_vis = SlimeVisualizer(env.W_pixels, env.H_pixels, **params_visualizer)
    actions = [4, 5]
    ACTION_NUM = len(params["actions"])
    AGENTS_NUM = env.cluster_learners + env.scatter_learners

    start_time = time.time()
    for ep in tqdm(range(1, EPISODES + 1), desc="Episode"):
        env.reset()
        for tick in tqdm(range(params['episode_ticks']), desc="Tick", leave=False):
            for agent in env.agent_iter(max_iter=AGENTS_NUM):
                observation, reward, _ , _, info = env.last(agent)
                action = np.random.randint(0, ACTION_NUM)
                #action = 1
                #action = 2
                #action = random.choice(actions)
                env.step(action)
            env_vis.render(
                env.patches,
                env.learners,
                # For debugging
                env.fov,
                env.ph_fov
            )
            #breakpoint()
        env.avg_cluster()

    print("Total time = ", time.time() - start_time)
    env.close()

if __name__ == "__main__":
    #PARAMS_FILE = "multi-agent-env-params.json"
    main()
    #cProfile.run("main()", "possible_refactor_min.prof")
