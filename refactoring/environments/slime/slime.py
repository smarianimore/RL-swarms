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

    def __init__(self,
                 seed,
                 render_mode: Optional[str] = None,
                 **kwargs):
        """
        :param population:          Controls the number of non-learning slimes (= green turtles)
        :param sniff_threshold:     Controls how sensitive slimes are to pheromone (higher values make slimes less
                                    sensitive to pheromone)—unclear effect on learning, could be negligible
        :param diffuse_area         Controls the diffusion radius
        :param diffuse_mode         Controls in which order patches with pheromone to diffuse are visited:
                                        'simple' = Python-dependant (dict keys "ordering")
                                        'rng' = random visiting
                                        'sorted' = diffuse first the patches with more pheromone
                                        'filter' = do not re-diffuse patches receiving pheromone due to diffusion
                                        'cascade' = step-by-step diffusion within 'diffuse_area'
        :param follow_mode          Controls how non-learning agents follow pheromone:
                                        'det' = follow greatest pheromone
                                        'prob' = follow greatest pheromone probabilistically (pheromone strength as weight)
        :param smell_area:          Controls the radius of the square area sorrounding the turtle whithin which it smells pheromone
        :param lay_area:            Controls the radius of the square area sorrounding the turtle where pheromone is laid
        :param lay_amount:          Controls how much pheromone is laid
        :param evaporation:         Controls how much pheromone evaporates at each step
        :param cluster_threshold:   Controls the minimum number of slimes needed to consider an aggregate within
                                    cluster-radius a cluster (the higher the more difficult to consider an aggregate a
                                    cluster)—the higher the more difficult to obtain a positive reward for being within
                                    a cluster for learning slimes
        :param cluster_radius:      Controls the range considered by slimes to count other slimes within a cluster (the
                                    higher the easier to form clusters, as turtles far apart are still counted together)
                                    —the higher the easier it is to obtain a positive reward for being within a cluster
                                    for learning slimes
        :param rew:                 Base reward for being in a cluster
        :param penalty:             Base penalty for not being in a cluster
        :param episode_ticks:       Number of ticks for episode termination
        :param W:                   Window width in # patches
        :param H:                   Window height in # patches
        :param PATCH_SIZE:          Patch size in pixels
        :param TURTLE_SIZE:         Turtle size in pixels
        :param FPS:                 Rendering FPS
        :param SHADE_STRENGTH:      Strength of color shading for pheromone rendering (higher -> brighter color)
        :param SHOW_CHEM_TEXT:      Whether to show pheromone amount on patches (when >= sniff-threshold)
        :param CLUSTER_FONT_SIZE:   Font size of cluster number (for overlapping agents)
        :param CHEMICAL_FONT_SIZE:  Font size of phermone amount (if SHOW_CHEM_TEXT is true)
        :param render_mode:
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
        self.reward_type = kwargs['reward_type']
        self.normalize_rewards = kwargs['normalize_rewards']
        self.reward = kwargs['rew']
        self.penalty = kwargs['penalty']
        self.episode_ticks = kwargs['episode_ticks']

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
        # DOC {(x,y): [(x,y), ..., (x,y)]} pre-computed diffusion area for each patch, including itself
        if self.diffuse_mode == "cascade":
            assert isinstance(self.diffuse_area, int), "Error: diffuse_area must be an int"
            self.diffuse_patches = self._find_neighbours_cascade(self.diffuse_area)
        elif self.diffuse_mode in ("rng", "sorted", "filter", "rng-filter"):
            assert isinstance(self.diffuse_area, int), "Error: diffuse_area must be an int"
            self.diffuse_patches = self._find_neighbours(self.diffuse_area)

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

        self.REWARD_MAX = self.reward + (((self.cluster_learners - 1) / self.cluster_threshold) * (self.reward ** 2))

        #Different from AECEnv attribute self.rewards - only keeps last step rewards
        #self.rewards_cust = {i: [] for i in range(self.population, pop_tot)}
        #self.cluster_ticks = {i: 0 for i in range(self.population, pop_tot)}
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(self.population, pop_tot)))
        )

    def _field_of_view(self, n_patches):
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

    def _find_neighbours_cascade(self, area: int):
        """
        For each patch, find neighbouring patches within square radius 'area', 1 step at a time
        (visiting first 1-hop patches, then 2-hops patches, and so on)
        """

        neighbours = {}
        
        for p in self.patches:
            neighbours[p] = []
            for ring in range(area):
                for x in range(p[0] + (ring * self.patch_size), p[0] + ((ring + 1) * self.patch_size) + 1,
                               self.patch_size):
                    for y in range(p[1] + (ring * self.patch_size), p[1] + ((ring + 1) * self.patch_size) + 1,
                                   self.patch_size):
                        if (x, y) not in neighbours[p]:
                            neighbours[p].append((x, y))
                for x in range(p[0] + (ring * self.patch_size), p[0] - ((ring + 1) * self.patch_size) - 1,
                               -self.patch_size):
                    for y in range(p[1] + (ring * self.patch_size), p[1] - ((ring + 1) * self.patch_size) - 1,
                                   -self.patch_size):
                        if (x, y) not in neighbours[p]:
                            neighbours[p].append((x, y))
                for x in range(p[0] + (ring * self.patch_size), p[0] + ((ring + 1) * self.patch_size) + 1,
                               self.patch_size):
                    for y in range(p[1] + (ring * self.patch_size), p[1] - ((ring + 1) * self.patch_size) - 1,
                                   -self.patch_size):
                        if (x, y) not in neighbours[p]:
                            neighbours[p].append((x, y))
                for x in range(p[0] + (ring * self.patch_size), p[0] - ((ring + 1) * self.patch_size) - 1,
                               -self.patch_size):
                    for y in range(p[1] + (ring * self.patch_size), p[1] + ((ring + 1) * self.patch_size) + 1,
                                   self.patch_size):
                        if (x, y) not in neighbours[p]:
                            neighbours[p].append((x, y))
            neighbours[p] = [self._wrap(x, y) for (x, y) in neighbours[p]]

        return neighbours

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

        # Non dovrei calcolare il reward e le osservazioni dopo aver fatto un'azione???
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
            #self.patches, self.learners[self.agent] = self.walk(self.patches, self.learners[self.agent])
            #self.patches, self.learners[self.agent] = self.walk2(self.patches, self.learners[self.agent])
            #self.patches = self.lay_pheromone(self.patches, self.learners[self.agent]['pos'])
            self.do_action0()
            self.do_action1()
        elif action == 5:   # Follow pheromone and Lay pheromone 
            #max_pheromone, max_coords = self._find_max_pheromone(self.learners[self.agent]['pos'])
            #max_pheromone, max_coords, max_ph_dir = self._find_max_pheromone2(
            #    self.learners[self.agent],
            #    self.observations[str(self.agent)]       
            #)
            #if max_pheromone >= self.sniff_threshold:
            #    #self.patches = self.follow_pheromone(self.patches, max_coords, self.learners[self.agent])
            #    self.patches, self.learners[self.agent] = self.follow_pheromone2(
            #        self.patches,
            #        max_coords,
            #        max_ph_dir,
            #        self.learners[self.agent]
            #    )
            #else:
            #    #self.patches, self.learners[self.agent] = self.walk(self.patches, self.learners[self.agent])
            #    self.patches, self.learners[self.agent] = self.walk2(self.patches, self.learners[self.agent])
            #self.patches = self.lay_pheromone(self.patches, self.learners[self.agent]['pos'])
            self.do_action2()
            self.do_action1()
        else:
            raise ValueError("Action out of range!")

        if self._agent_selector.is_last():
            for ag in self.agents:
                self.rewards[ag] = self.rewards_cust[self.agent_name_mapping[ag]][-1]
            #if len(self.turtles) > 0:
            #    self.turtles, self.patches = self.move(self.turtles, self.patches)
            #self.patches = self._diffuse(self.patches)
            #self.patches = self._diffuse2(self.patches)
            #self.patches = self._evaporate(self.patches)
            self.patches = self._diffuse_and_evaporate(self.patches)
        else:
            self._clear_rewards()
            
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[str(self.agent)] = 0
        self._accumulate_rewards()
        
    """
    def move(self, turtles, patches):
        for turtle in turtles:
            pos = turtles[turtle]['pos']
            t = turtles[turtle]
            max_pheromone, max_coords = self._find_max_pheromone(patches, pos)

            if max_pheromone >= self.sniff_threshold:
                patches = self.follow_pheromone(patches, max_coords, t, turtle)
            else:
                patches, turtle = self.walk(patches, t, turtle)

            patches = self.lay_pheromone(patches, turtles[turtle]['pos'])

        return turtles, patches 
    """

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

        #if self.reward_type == "cluster":
        if self.learners[self.agent]["mode"] == 'c':
            cluster_ticks, rewards_cust, cur_reward = self.reward_cluster_and_time_punish_time(
                cluster_ticks,
                rewards_cust,
                cluster
            )
        #elif self.reward_type == "scatter":
        elif self.learners[self.agent]["mode"] == 's':
            cluster_ticks, rewards_cust, cur_reward = self.reward_scatter_and_time_punish_time(
                cluster_ticks,
                rewards_cust,
                cluster
            )
        
        if self.obs_type == "paper":
            #_, max_coords = self._find_max_pheromone(self.learners[self.agent]['pos'])
            #observations = np.array(max_coords)
            #observations = self._get_obs(self.learners[self.agent]['pos'])
            observations = self._get_obs2(self.learners[self.agent])
        elif self.obs_type == "variation1":
            observations = self._get_obs3(self.learners[self.agent])
        elif self.obs_type == "variation2":
            chemical = self._check_chemical(self.agent)
            observations = np.array([cluster >= self.cluster_threshold, chemical])

        return observations, cluster_ticks, rewards_cust

    def _get_obs(self, pos):
        """
        This method return the observation give by the env.
        The array indicates the pheromone values in the 8 patches around the agent.
        """
        field_of_view = [
            self._wrap(r, c)
            for r in range(pos[0] - self.patch_size, pos[0] + 2 * self.patch_size, self.patch_size)
            for c in range(pos[1] - self.patch_size, pos[1] + 2 * self.patch_size, self.patch_size)
        ]
        field_of_view.remove(pos)
        obs = np.array([self.patches[f]["chemical"] for f in field_of_view])
        return obs 

    def _get_obs2(self, agent):
        f, _ = self._get_new_positions(self.ph_fov, agent)
        obs = np.array([self.patches[tuple(i)]["chemical"] for i in f])
        return obs

    def _get_obs3(self, agent):
        f, _ = self._get_new_positions(self.ph_fov, agent)
        obs = [self.patches[tuple(i)]["chemical"] for i in f]
        obs.append(self.patches[agent["pos"]]["chemical"])
        return np.array(obs)

    def do_action0(self):
        #self.patches, self.learners[self.agent] = self.walk(self.patches, self.learners[self.agent])
        self.patches, self.learners[self.agent] = self.walk2(self.patches, self.learners[self.agent])

    def do_action1(self):
        self.patches = self.lay_pheromone(self.patches, self.learners[self.agent]['pos'])

    def do_action2(self):
        #max_pheromone, max_coords = self._find_max_pheromone(self.learners[self.agent]['pos'])
        if self.obs_type == "paper":
            max_pheromone, max_coords, max_ph_dir = self._find_max_pheromone2(
                self.learners[self.agent],
                self.observations[str(self.agent)]       
            )
            if max_pheromone >= self.sniff_threshold:
                #self.patches = self.follow_pheromone(self.patches, max_coords, self.learners[self.agent])
                self.patches, self.learners[self.agent] = self.follow_pheromone2(
                    self.patches,
                    max_coords,
                    max_ph_dir,
                    self.learners[self.agent]
                )
            else:
                #self.patches, self.learners[self.agent] = self.walk(self.patches, self.learners[self.agent])
                #self.patches, self.learners[self.agent] = self.walk2(self.patches, self.learners[self.agent])
                self.do_action0()
        elif self.obs_type == "variation1":
            max_pheromone, max_coords, max_ph_dir = self._find_max_pheromone3(
                self.learners[self.agent],
                self.observations[str(self.agent)]       
            )
            if max_pheromone >= self.sniff_threshold:
                #self.patches = self.follow_pheromone(self.patches, max_coords, self.learners[self.agent])
                if max_coords != self.learners[self.agent]["pos"]:
                    self.patches, self.learners[self.agent] = self.follow_pheromone2(
                        self.patches,
                        max_coords,
                        max_ph_dir,
                        self.learners[self.agent]
                    )
            else:
                #self.patches, self.learners[self.agent] = self.walk(self.patches, self.learners[self.agent])
                #self.patches, self.learners[self.agent] = self.walk2(self.patches, self.learners[self.agent])
                self.do_action0()

    def do_action3(self):
        #max_pheromone, max_coords = self._find_max_pheromone(self.learners[self.agent]['pos'])
        if np.any(self.observations[str(self.agent)] >= self.sniff_threshold):
            ph_pos, ph_dir = self._find_non_max_pheromone(self.learners[self.agent], self.observations[str(self.agent)])
            self.patches, self.learners[self.agent] = self.avoid_pheromone(
                self.patches,
                ph_pos,
                ph_dir,
                self.learners[self.agent]
            )
        #if max_pheromone >= self.sniff_threshold:
        #    #self.patches = self.run_away_pheromone(self.patches, max_coords, self.learners[self.agent])
        else:
            #self.patches, self.learners[self.agent] = self.walk(self.patches, self.learners[self.agent])
            self.do_action0()

    def convert_observation(self, obs):
        """
        This method returns the conversion of the observation to an integer.
        It's useful for IQL.
        """
        if self.obs_type == "paper":
            if np.unique(obs).shape[0] == 1:
                obs_id = np.random.randint(8)
            else:
                obs_id = obs.argmax().item()
        elif self.obs_type == "variation_1":
            obs_id = int(f"{obs[0].astype(np.uint8)}{obs[1].astype(np.uint8)}", 2)
        return obs_id
    
    def convert_observation2(self, obs):
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

    def _diffuse(self, patches):
        """
        Diffuses pheromone from each patch to nearby patches controlled through self.diffuse_area patches in a way
        controlled through self.diffuse_mode:
            'simple' = Python-dependant (dict keys "ordering")
            'rng' = random visiting
            'sorted' = diffuse first the patches with more pheromone
            'filter' = do not re-diffuse patches receiving pheromone due to diffusion
        """
        n_size = len(self.diffuse_patches[list(patches.keys())[0]])  # same for every patch
        patch_keys = list(patches.keys())
        
        if self.diffuse_mode == 'rng':
            random.shuffle(patch_keys)
        elif self.diffuse_mode == 'sorted':
            patch_list = list(patches.items())
            patch_list = sorted(patch_list, key=lambda t: t[1]['chemical'], reverse=True)
            patch_keys = [t[0] for t in patch_list]
        elif self.diffuse_mode == 'filter':
            patch_keys = [k for k in patches if patches[k]['chemical'] > 0]
        elif self.diffuse_mode == 'rng-filter':
            patch_keys = [k for k in patches if patches[k]['chemical'] > 0]
            random.shuffle(patch_keys)
        
        for patch in patch_keys:
            p = patches[patch]['chemical']
            ratio = p / n_size
            
            if p > 0:
                diffuse_keys = self.diffuse_patches[patch][:]
                
                for n in diffuse_keys:
                    patches[n]['chemical'] += ratio
                
                patches[patch]['chemical'] = ratio

        return patches
    
    def _diffuse2(self, patches):
        """
        This diffuse method use a gaussian filter for the process.
        This is a kind of parallel diffusion.
        """
        grid = np.array([patches[p]["chemical"] for p in patches.keys()]).reshape((self.W, self.H))
        grid = gaussian_filter(grid, sigma=self.diffuse_area, mode="wrap")
        grid = grid.flatten()
        for p, g in zip(patches, grid):
            patches[p]['chemical'] = g
        return patches

    def _evaporate(self, patches):
        """
        Evaporates pheromone from each patch according to param self.evaporation
        """
        for patch in patches.keys():
            #if patches[patch]['chemical'] > 0:
            patches[patch]['chemical'] *= self.evaporation

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
        """
        Action 0: move in random direction (8 sorrounding cells)
        """      
        choice = [self.patch_size, -self.patch_size, 0]
        x, y = turtle['pos']
        patches[turtle['pos']]['turtles'].remove(self.agent)
        x_rnd = np.random.choice(choice)
        y_rnd = np.random.choice(choice)
        x2, y2 = x + x_rnd, y + y_rnd
        x2, y2 = self._wrap(x2, y2)
        
        turtle['pos'] = (x2, y2)
        patches[turtle['pos']]['turtles'].append(self.agent)

        return patches, turtle
    
    def walk2(self, patches, turtle):
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

    def run_away_pheromone(self, patches, ph_coords, turtle):
        """
        Action 3: don't follow/avoid the pheromone.
        """
        x, y = turtle['pos']
        patches[turtle['pos']]['turtles'].remove(self.agent)
        if ph_coords[0] > x and ph_coords[1] > y:
            x -= self.patch_size
            y -= self.patch_size
        elif ph_coords[0] < x and ph_coords[1] < y:
            x += self.patch_size
            y += self.patch_size
        elif ph_coords[0] > x and ph_coords[1] < y:
            x -= self.patch_size
            y += self.patch_size
        elif ph_coords[0] < x and ph_coords[1] > y:
            x += self.patch_size
            y -= self.patch_size
        elif ph_coords[0] == x and ph_coords[1] < y:
            choices = [self.patch_size, -self.patch_size]
            x += random.choice(choices)
            y += self.patch_size
        elif ph_coords[0] == x and ph_coords[1] > y:
            choices = [self.patch_size, -self.patch_size]
            x += random.choice(choices)
            y -= self.patch_size
        elif ph_coords[0] > x and ph_coords[1] == y:  
            choices = [self.patch_size, -self.patch_size]
            x -= self.patch_size
            y += random.choice(choices)
        elif ph_coords[0] < x and ph_coords[1] == y:   
            choices = [self.patch_size, -self.patch_size]
            x += self.patch_size
            y += random.choice(choices)
        else:  # my patch
            choices = [self.patch_size, -self.patch_size]
            x += random.choice(choices)
            y += random.choice(choices)
        x, y = self._wrap(x, y)
        turtle['pos'] = (x, y)
        patches[turtle['pos']]['turtles'].append(self.agent)

        return patches
    
    def avoid_pheromone(self, patches, ph_coords, ph_dir, turtle):
        patches[turtle['pos']]['turtles'].remove(self.agent)
        turtle["pos"] = ph_coords
        patches[turtle['pos']]['turtles'].append(self.agent)
        turtle["dir"] = ph_dir
        return patches, turtle

    def follow_pheromone(self, patches, ph_coords, turtle):
        """
        Action 2: move turtle towards greatest pheromone found
        """
        x, y = turtle['pos']
        patches[turtle['pos']]['turtles'].remove(self.agent)
        if ph_coords[0] > x and ph_coords[1] > y:  # top right
            x += self.patch_size
            y += self.patch_size
        elif ph_coords[0] < x and ph_coords[1] < y:  # bottom left
            x -= self.patch_size
            y -= self.patch_size
        elif ph_coords[0] > x and ph_coords[1] < y:  # bottom right
            x += self.patch_size
            y -= self.patch_size
        elif ph_coords[0] < x and ph_coords[1] > y:  # top left
            x -= self.patch_size
            y += self.patch_size
        elif ph_coords[0] == x and ph_coords[1] < y:  # below me
            y -= self.patch_size
        elif ph_coords[0] == x and ph_coords[1] > y:  # above me
            y += self.patch_size
        elif ph_coords[0] > x and ph_coords[1] == y:  # right
            x += self.patch_size
        elif ph_coords[0] < x and ph_coords[1] == y:  # left
            x -= self.patch_size
        else:  # my patch
            pass
        x, y = self._wrap(x, y)
        turtle['pos'] = (x, y)
        patches[turtle['pos']]['turtles'].append(self.agent)

        return patches
    
    def follow_pheromone2(self, patches, ph_coords, ph_dir, turtle):
        patches[turtle['pos']]['turtles'].remove(self.agent)
        turtle["pos"] = ph_coords
        patches[turtle['pos']]['turtles'].append(self.agent)
        turtle["dir"] = ph_dir
        return patches, turtle

    def _find_max_pheromone(self, pos):
        """
        Find where the maximum pheromone level is within a square controlled by self.smell_area centred in 'pos'.
        Following pheromone modeis controlled by param self.follow_mode:
            'det' = follow greatest pheromone
            'prob' = follow greatest pheromone probabilistically (pheromone strength as weight)
        """
        if self.follow_mode == "prob":
            population = [k for k in self.smell_patches[pos]]
            weights = [self.patches[k]['chemical'] for k in self.smell_patches[pos]]
            if all([w == 0 for w in weights]):
                winner = population[np.random.choice(len(population))]
            else:
                winner = random.choices(population, weights=weights, k=1)[0]
            max_ph = self.patches[winner]['chemical']
        else:
            max_ph = -1
            max_pos = [pos]
            for p in self.smell_patches[pos]:
                chem = self.patches[p]['chemical']
                if chem > max_ph:
                    max_ph = chem
                    max_pos = [p]
                elif chem == max_ph:
                    max_pos.append(p)
            winner = max_pos[np.random.choice(len(max_pos))]

        return max_ph, winner

    def _find_max_pheromone2(self, agent, obs):
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

    def _find_max_pheromone3(self, agent, obs):
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
        ids = np.where(obs >= self.sniff_threshold)[0]
        if ids.shape[0] == obs.shape[0]:
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

    #def avg_cluster(self):
    #    """
    #    Record the cluster size. It's a fuzzy computation.
    #    """
    #    cluster_sizes = []  # registra la dim. dei cluster
    #    for l in self.learners:
    #        cluster = []  # tiene conto di quali turtle sono in quel cluster
    #        for p in self.cluster_patches[self.learners[l]['pos']]:
    #            for t in self.patches[p]['turtles']:
    #                cluster.append(t)
    #        cluster.sort()
    #        if cluster not in cluster_sizes:
    #            cluster_sizes.append(cluster)

    #    # cleaning process: confornta i cluster (nello stesso episodio) e se ne trova 2 con più del 90% di turtle uguali ne elimina 1
    #    for cluster in cluster_sizes:
    #        for cl in cluster_sizes:
    #            if cl != cluster:
    #                intersection = list(set(cluster) & set(cl))
    #                if len(intersection) > len(cluster) * 0.90:
    #                    cluster_sizes.remove(cl)

    #    # calcolo avg_cluster_size
    #    somma = 0
    #    for cluster in cluster_sizes:
    #        somma += len(cluster)
    #    avg_cluster_size = somma / len(cluster_sizes)
    #    return avg_cluster_size
    
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

    def avg_cluster2(self):
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

    # not a real reward function
    def test_reward(self, current_agent):  # trying to invert rewards process, GOAL: check any strange behaviour
        """
        :return: the reward
        """
        self.agent = current_agent
        chem = 0
        for p in self.patches.values():
            if self.agent in p['turtles']:
                chem = p['chemical']
        if chem >= 5:
            cur_reward = -1000
        else:
            cur_reward = 100

        self.rewards_cust[self.agent].append(cur_reward)
        return cur_reward

    def reward_cluster_punish_time(self, current_agent):  # DOC NetLogo rewardFunc7
        """
        Reward is (positve) proportional to cluster size (quadratic) and (negative) proportional to time spent outside
        clusters
        """
        self.agent = current_agent
        cluster = self._compute_cluster(self.agent)
        if cluster >= self.cluster_threshold:
            self.cluster_ticks[self.agent] += 1

        cur_reward = ((cluster ^ 2) / self.cluster_threshold) * self.reward + (
                ((self.episode_ticks - self.cluster_ticks[self.agent]) / self.episode_ticks) * self.penalty)

        self.rewards_cust[self.agent].append(cur_reward)
        return cur_reward

    def reward_cluster_and_time_punish_time(self, cluster_ticks, rewards_cust, cluster):
        """
        The clustering reward used in the article.
        """
        if cluster >= self.cluster_threshold:
            cluster_ticks[self.agent] += 1

        cur_reward = (cluster_ticks[self.agent] / self.episode_ticks) * self.reward + \
                     (cluster / self.cluster_threshold) * (self.reward ** 2) + \
                     (((self.episode_ticks - cluster_ticks[self.agent]) / self.episode_ticks) * self.penalty)
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

        cur_reward = (cluster_ticks[self.agent] / self.episode_ticks) * self.penalty + \
                     (cluster / self.cluster_threshold) * (self.penalty ** 2) + \
                     (((self.episode_ticks - cluster_ticks[self.agent]) / self.episode_ticks) * self.reward)
        if self.normalize_rewards:
            cur_reward = round(cur_reward / self.reward, 4)

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

    def get_neighborood_chemical(self, agent, as_vectors=False):
        agent_pos = self.learners[agent]["pos"]
        smell_patches = self.smell_patches[agent_pos]
        
        output_mask = []
        for patch in smell_patches:
            output_mask.append(self.patches[patch]["chemical"] - self.patches[agent_pos]["chemical"]) if as_vectors else output_mask.append(self.patches[patch]["chemical"])

        return np.array([output_mask], dtype=np.float32)


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
        "cluster_learners": 0,
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
        "reward_type": "scatter",
        "normalize_rewards": True,
        "rew": 100,
        "penalty": -1,
        #"episode_ticks": 500,
        "episode_ticks": 500,
        #"W": 16,
        "W": 22,
        #"H": 16,
        "H": 22,
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
    SEED = 2
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
            #env_vis.render(
            #    env.patches,
            #    env.learners,
            #    # For debugging
            #    env.fov,
            #    env.ph_fov
            #)
            #breakpoint()
        env.avg_cluster2()

    print("Total time = ", time.time() - start_time)
    env.close()

if __name__ == "__main__":
    #PARAMS_FILE = "multi-agent-env-params.json"
    main()
    #cProfile.run("main()", "possible_refactor_min.prof")
