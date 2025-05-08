import random
import sys
from typing import Optional
import time
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter

from gymnasium.spaces import Discrete, MultiBinary, Box

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import ObsType

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
        :param sniff_threshold:     Controls how sensitive slimes are to pheromone (higher values make slimes less
                                    sensitive to pheromone)—unclear effect on learning, could be negligible
        :param diffuse_area         Controls the diffusion radius
        :param follow_mode          Controls how non-learning agents follow pheromone:
                                        'det' = follow greatest pheromone
                                        'prob' = follow greatest pheromone probabilistically (pheromone strength as weight)
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
        
        self.cluster_learners = kwargs['cluster_learners'] 
        self.scatter_learners = kwargs['scatter_learners'] 
        self.sniff_threshold = kwargs['sniff_threshold']
        self.diffuse_area = kwargs['diffuse_area']
        self.diffuse_radius = kwargs['diffuse_radius']
        self.lay_area = kwargs['lay_area']
        self.lay_amount = kwargs['lay_amount']
        self.evaporation = kwargs['evaporation']
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

        pop_tot = self.cluster_learners + self.scatter_learners
        self.possible_agents = [str(i) for i in range(pop_tot)]  # DOC learning agents IDs
        self._agent_selector = agent_selector(self.possible_agents)
        self.agent = self._agent_selector.reset()

        n_coords = len(self.coords)
        # patches-own [chemical] - amount of pheromone in each patch
        self.patches = {
            self.coords[i]: {
                'id': i,
                'chemical_0': 0.0,
                'chemical_1': 0.0,
                'turtles': []
            }
            for i in range(n_coords)
        }
        # create learners turtle
        self.learners = {
            i: {
                'pos': self.coords[np.random.randint(n_coords)],
                'dir': np.random.randint(self.N_DIRS), 
                'mode': 'c' if i < self.cluster_learners else 's'
            }
            for i in range(self.cluster_learners + self.scatter_learners)
        }
        for l in self.learners:
            self.patches[self.learners[l]['pos']]['turtles'].append(l)  # DOC id of learner turtles

        # pre-compute relevant structures to speed-up computation during rendering steps
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
        
        assert kwargs['obs_type'] in ("paper", "variation1", "variation2"), "Error!"
        self.obs_type = kwargs['obs_type']
        # DOC obervation is an array of 8 real elements.
        # This array indicates the pheromone values in the 8 patches around the agent.
        
        if self.obs_type == "paper":
            self._observation_spaces = {
                a: Box(low=0.0, high=np.inf, shape=(self.sniff_patches * 2,), dtype=np.float32)
                for a in self.possible_agents
            }
        elif self.obs_type == "variation1":
            pass

        #self.REWARD_MAX = self.cluster_reward + (((self.cluster_learners - 1) / self.cluster_threshold) * (self.cluster_reward ** 2))

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(pop_tot)))
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

    def _compute_cluster_for_clustering(self, current_agent):
        # Versione 1: qui considero il cluster essere formato solo da agenti del mio stesso tipo. 
        #cluster = -1
        #for p in self.cluster_patches[self.learners[current_agent]['pos']]:
        #    for t in self.patches[p]['turtles']:
        #        if self.learners[current_agent]['mode'] == self.learners[t]['mode']:
        #            cluster += 1
        
        clusters = {'ph_1': 0, 'ph_2': 0}
        for p in self.cluster_patches[self.learners[current_agent]['pos']]:
            for t in self.patches[p]['turtles']:
                if t != current_agent and self.learners[t]['mode'] == 'c':
                    clusters['ph_1'] += 1
                elif t != current_agent and self.learners[t]['mode'] == 's':
                    clusters['ph_2'] += 1

        return clusters

    #def _compute_cluster_for_clustering(self, current_agent):
    #    # Versione 1: qui considero il cluster essere formato solo da agenti del mio stesso tipo. 
    #    #cluster = -1
    #    #for p in self.cluster_patches[self.learners[current_agent]['pos']]:
    #    #    for t in self.patches[p]['turtles']:
    #    #        if self.learners[current_agent]['mode'] == self.learners[t]['mode']:
    #    #            cluster += 1
    #    
    #    clusters = {'ph_1': 0, 'ph_2': 0}
    #    for p in self.cluster_patches[self.learners[current_agent]['pos']]:
    #        for t in self.patches[p]['turtles']:
    #            if t != current_agent and self.learners[t]['mode'] == 'c':
    #                clusters['ph_1'] += 1
    #            elif t != current_agent and self.learners[t]['mode'] == 's':
    #                clusters['ph_2'] += 1
    #    if self.learners[current_agent]['mode'] == 'c':
    #        cluster = clusters['ph_1'] - clusters['ph_2']
    #    elif self.learners[current_agent]['mode'] == 's':
    #        cluster = clusters['ph_2'] - clusters['ph_1']

    #    return cluster
    
    def _compute_cluster_for_scattering(self, current_agent):
        cluster = -1
        
        #for p in self.cluster_patches[self.learners[current_agent]['pos']]:
        #    for t in self.patches[p]['turtles']:
        #        if self.learners[current_agent]['mode'] == self.learners[t]['mode']:
        #            cluster += 1

        for p in self.cluster_patches[self.learners[current_agent]['pos']]:
            cluster += len(self.patches[p]['turtles'])

        return cluster

    #def _compute_cluster(self, current_agent):
    #    """
    #    Checks whether the learner turtle is within a cluster, given 'cluster_radius' and 'cluster_threshold'
    #    """
    #    #cluster = 0
    #    cluster = -1
    #    if self.learners[self.agent]["mode"] == 's':
    #        for p in self.cluster_patches[self.learners[current_agent]['pos']]:
    #            cluster += len(self.patches[p]['turtles'])

    #    # Versione 2: qui ho una penalità, calcolo i cluster "corretto" e sottraggo quello "sbagliato". 
    #    #elif self.learners[self.agent]["mode"] == 'c':
    #    #    clusters = {'ph_1': 0, 'ph_2': 0}
    #    #    for p in self.cluster_patches[self.learners[current_agent]['pos']]:
    #    #        for t in self.patches[p]['turtles']:
    #    #            if t != current_agent and self.learners[t]['mode'] == 'c':
    #    #                clusters['ph_1'] += 1
    #    #            elif t != current_agent and self.learners[t]['mode'] == 's':
    #    #                clusters['ph_2'] += 1
    #    #    if self.learners[current_agent]['mode'] == 'c':
    #    #        cluster = clusters['ph_1'] - clusters['ph_2']
    #    #    elif self.learners[current_agent]['mode'] == 's':
    #    #        cluster = clusters['ph_2'] - clusters['ph_1']


    #    #breakpoint()
    #    return cluster

    def reward_cluster_double_ph(self, cluster_ticks, rewards_cust, clusters):
        """
        The clustering reward used in the article.
        """
        # Versione del reward senza penalità nel caso in cui l'agente faccia clustering con il gruppo sbagliato
        #breakpoint()
        if self.learners[self.agent]['mode'] == 'c':
            cluster = clusters['ph_1']
            intruders = -clusters['ph_2']
        elif self.learners[self.agent]['mode'] == 's':
            cluster = clusters['ph_2']
            intruders = -clusters['ph_1']

        if cluster >= self.cluster_threshold:
            cluster_ticks[self.agent] += 1

        cur_reward = (cluster_ticks[self.agent] / self.episode_ticks) * self.cluster_reward + \
                     (cluster / self.cluster_threshold) * (self.cluster_reward ** 2) + \
                     (((self.episode_ticks - cluster_ticks[self.agent]) / self.episode_ticks) * self.cluster_penalty) + \
                     intruders * ((self.cluster_reward / 2) ** 2)

        #if self.normalize_rewards:
        #    cur_reward = round(cur_reward / self.REWARD_MAX, 4)


        rewards_cust[self.agent].append(cur_reward)
        return cluster_ticks, rewards_cust, cur_reward

    #def reward_cluster_double_ph(self, cluster_ticks, rewards_cust, cluster):
    #    """
    #    The clustering reward used in the article.
    #    """
    #    # Versione del reward senza penalità nel caso in cui l'agente faccia clustering con il gruppo sbagliato
    #    #breakpoint()
    #    if cluster >= self.cluster_threshold:
    #        cluster_ticks[self.agent] += 1

    #    cur_reward = (cluster_ticks[self.agent] / self.episode_ticks) * self.cluster_reward + \
    #                 (cluster / self.cluster_threshold) * (self.cluster_reward ** 2) + \
    #                 (((self.episode_ticks - cluster_ticks[self.agent]) / self.episode_ticks) * self.cluster_penalty)

    #    #if self.normalize_rewards:
    #    #    cur_reward = round(cur_reward / self.REWARD_MAX, 4)


    #    rewards_cust[self.agent].append(cur_reward)
    #    return cluster_ticks, rewards_cust, cur_reward
    
    def reward_scatter_double_ph(self, cluster_ticks, rewards_cust, cluster):
        """
        The scattering reward used in the article.
        """
        if cluster >= self.cluster_threshold:
            cluster_ticks[self.agent] += 1

        cur_reward = (cluster_ticks[self.agent] / self.episode_ticks) * self.scatter_penalty - \
                     (cluster / self.cluster_threshold) * (self.scatter_penalty ** 2) + \
                     (((self.episode_ticks - cluster_ticks[self.agent]) / self.episode_ticks) * self.scatter_reward)

        #if self.normalize_rewards and self.scatter_reward != 0:
        #    cur_reward = round(cur_reward / self.scatter_reward, 4)

        rewards_cust[self.agent].append(cur_reward)
        return cluster_ticks, rewards_cust, cur_reward
    
    def _get_obs2(self, agent):
        f, _ = self._get_new_positions(self.ph_fov, agent)

        obs_ph_0 = [self.patches[tuple(i)]["chemical_0"] for i in f]
        obs_ph_1 = [self.patches[tuple(i)]["chemical_1"] for i in f]
        obs = np.array(obs_ph_0 + obs_ph_1)
        return obs
    
    def process_agent(self, cluster_ticks, rewards_cust):
        """
        In this methods we compute the agent's reward and it's observation.
        """
        # only cluster
        #cluster = self._compute_cluster_for_clustering(self.agent)
        clusters = self._compute_cluster_for_clustering(self.agent)
        cluster_ticks, rewards_cust, cur_reward = self.reward_cluster_double_ph(
            cluster_ticks,
            rewards_cust,
            clusters
            #cluster
        )
        
        # only scatter
        #cluster = self._compute_cluster_for_scattering(self.agent)
        #cluster_ticks, rewards_cust, cur_reward = self.reward_scatter_double_ph(
        #    cluster_ticks,
        #    rewards_cust,
        #    cluster
        #)

        # mixed
        #if self.learners[self.agent]["mode"] == 'c':
        #    cluster_ticks, rewards_cust, cur_reward = self.reward_cluster_double_ph(
        #        cluster_ticks,
        #        rewards_cust,
        #        cluster
        #    )
        #elif self.learners[self.agent]["mode"] == 's':
        #    cluster_ticks, rewards_cust, cur_reward = self.reward_scatter_double_ph(
        #        cluster_ticks,
        #        rewards_cust,
        #        cluster
        #    )

        if self.obs_type == "paper":
            observations = self._get_obs2(self.learners[self.agent])
        elif self.obs_type == "variation1":
            observations = self._get_obs3(self.learners[self.agent])

        #breakpoint()
        return observations, cluster_ticks, rewards_cust

    def _walk2(self, patches, turtle):
        """
        Action 0: move in random direction (8 sorrounding cells)
        """      
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

    def do_action0(self):
        self.patches, self.learners[self.agent] = self._walk2(self.patches, self.learners[self.agent])

    def _lay_pheromone_0(self, patches, agent_pos):
        """
        Lay 'amount' pheromone in square 'area' centred in 'pos'
        """
        for p in self.lay_patches[agent_pos]:
            patches[p]['chemical_0'] += self.lay_amount
        
        return patches

    def do_action1(self):
        self.patches = self._lay_pheromone_0(self.patches, self.learners[self.agent]['pos'])

    def _lay_pheromone_1(self, patches, agent_pos):
        """
        Lay 'amount' pheromone in square 'area' centred in 'pos'
        """
        for p in self.lay_patches[agent_pos]:
                patches[p]['chemical_1'] += self.lay_amount
        
        return patches

    def do_action2(self):
        self.patches = self._lay_pheromone_1(self.patches, self.learners[self.agent]['pos'])
    
    def _find_max_pheromone2(self, agent, obs):
        """
        Following pheromone modeis controlled by param self.follow_mode:
            'det' = follow greatest pheromone
            'prob' = follow greatest pheromone probabilistically (pheromone strength as weight)
        """
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

    def _follow_pheromone2(self, patches, ph_coords, ph_dir, turtle):
        """
        Action 2: move turtle towards greatest pheromone found
        """
        patches[turtle['pos']]['turtles'].remove(self.agent)
        turtle["pos"] = ph_coords
        patches[turtle['pos']]['turtles'].append(self.agent)
        turtle["dir"] = ph_dir
        return patches, turtle

    def do_action3(self):
        if self.obs_type == "paper":
            max_pheromone, max_coords, max_ph_dir = self._find_max_pheromone2(
                self.learners[self.agent],
                self.observations[str(self.agent)][:self.sniff_patches]        
            )
            if max_pheromone >= self.sniff_threshold:
                self.patches, self.learners[self.agent] = self._follow_pheromone2(
                    self.patches,
                    max_coords,
                    max_ph_dir,
                    self.learners[self.agent]
                )
            else:
                self.do_action0()
        elif self.obs_type == "variation1":
            pass
    
    def do_action4(self):
        if self.obs_type == "paper":
            max_pheromone, max_coords, max_ph_dir = self._find_max_pheromone2(
                self.learners[self.agent],
                self.observations[str(self.agent)][self.sniff_patches:]        
            )
            if max_pheromone >= self.sniff_threshold:
                self.patches, self.learners[self.agent] = self._follow_pheromone2(
                    self.patches,
                    max_coords,
                    max_ph_dir,
                    self.learners[self.agent]
                )
            else:
                self.do_action0()
        elif self.obs_type == "variation1":
            pass

    def _find_non_max_pheromone(self, agent, obs):
        f, direction = self._get_new_positions(self.ph_fov, agent)
        ids = np.where(obs < self.sniff_threshold)[0]
        
        if ids.shape[0] == 0:
            idx = obs.argmin()
        else:
            idx = np.random.choice(ids)

        ph_pos = tuple(f[idx])
        
        if self.sniff_patches < self.N_DIRS:
            ph_dir = self._get_new_direction(self.sniff_patches, direction, idx)
        else:
            ph_dir = idx
            #ph_dir = idx % self.sniff_patches
        return ph_pos, ph_dir

    def _avoid_pheromone(self, patches, ph_coords, ph_dir, turtle):
        """
        Avoid the pheromone.
        """
        patches[turtle['pos']]['turtles'].remove(self.agent)
        turtle["pos"] = ph_coords
        patches[turtle['pos']]['turtles'].append(self.agent)
        turtle["dir"] = ph_dir
        return patches, turtle

    def do_action5(self):
        if np.any(self.observations[str(self.agent)] >= self.sniff_threshold):
            if self.obs_type == "paper":
                ph_pos, ph_dir = self._find_non_max_pheromone(
                    self.learners[self.agent], 
                    self.observations[str(self.agent)][:self.sniff_patches]        
                )
                self.patches, self.learners[self.agent] = self._avoid_pheromone(
                    self.patches,
                    ph_pos,
                    ph_dir,
                    self.learners[self.agent]
                )
            elif self.obs_type == "variation1":
                pass
        else:
            self.do_action0()
    
    def do_action6(self):
        if np.any(self.observations[str(self.agent)] >= self.sniff_threshold):
            if self.obs_type == "paper":
                ph_pos, ph_dir = self._find_non_max_pheromone(
                    self.learners[self.agent], 
                    self.observations[str(self.agent)][self.sniff_patches:]        
                )
                self.patches, self.learners[self.agent] = self._avoid_pheromone(
                    self.patches,
                    ph_pos,
                    ph_dir,
                    self.learners[self.agent]
                )
            elif self.obs_type == "variation1":
                pass
        else:
            self.do_action0()

    def _diffuse_and_evaporate(self, patches):
        """
        This diffuse method use a gaussian filter for the process.
        This is a kind of parallel diffusion.
        Evaporates pheromone from each patch according to param self.evaporation
        """
        #breakpoint()
        # Diffusion
        grid0 = np.array([patches[p]["chemical_0"] for p in patches.keys()]).reshape((self.W, self.H))
        grid1 = np.array([patches[p]["chemical_1"] for p in patches.keys()]).reshape((self.W, self.H))
        
        if self.diffuse_radius == 0:
            grid0 = gaussian_filter(grid0, sigma=self.diffuse_area, mode="wrap")
            grid1 = gaussian_filter(grid1, sigma=self.diffuse_area, mode="wrap")
        else:
            grid0 = gaussian_filter(grid0, sigma=self.diffuse_area, radius=self.diffuse_radius, mode="wrap")
            grid1 = gaussian_filter(grid1, sigma=self.diffuse_area, radius=self.diffuse_radius, mode="wrap")
        
        grid0 = grid0.flatten()
        grid1 = grid1.flatten()
        # Evaporation
        grid0 *= self.evaporation
        grid1 *= self.evaporation
        # Write values
        for p, g0, g1 in zip(patches, grid0, grid1):
            patches[p]['chemical_0'] = g0
            patches[p]['chemical_1'] = g1
        
        return patches

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
        
        if action == 0:     # Random walk
            self.do_action0()   
        elif action == 1:   # Lay pheromone 0
            self.do_action1()
        elif action == 2:   # Lay pheromone 1
            self.do_action2()
        elif action == 3:   # Follow pheromone 0
            self.do_action3()
        elif action == 4:   # Follow pheromone 1
            self.do_action4()
        elif action == 5:   # Avoid pheromone 0
            self.do_action5()
        elif action == 6:   # Avoid pheromone 1
            self.do_action6()
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

    def reset(self, seed=None, return_info=True, options=None):
        """
        Reset env.
        """
        # empty stuff
        pop_tot = self.cluster_learners + self.scatter_learners
        #Different from AECEnv attribute self.rewards - only keeps last step rewards
        self.rewards_cust = {i: [] for i in range(pop_tot)}
        self.cluster_ticks = {i: 0 for i in range(pop_tot)}
        
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
            self.patches[p]['chemical_0'] = 0.0
            self.patches[p]['chemical_1'] = 0.0
        
        if self.obs_type == "paper":
            self.observations = {
                a: np.zeros(self.sniff_patches * 2, dtype=np.float32)
                for a in self.agents
            }
        elif self.obs_type == "variation1":
            pass
        
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
    
    def convert_observation(self, obs):
        """
        This method returns the conversion of the observation to an integer.
        It's useful for IQL.
        """

        if self.obs_type == "paper":
            if np.unique(obs).shape[0] == 1:
                obs_id = np.random.randint(self.sniff_patches * 2)
            else:
                obs_id = obs.argmax() 
        elif self.obs_type == "variation1":
            pass
        
        return obs_id
    
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
        self.sniff_threshold = 0.0 #kwargs['sniff_threshold']
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
        self.ph_pos_font = pygame.font.SysFont("arial", self.chemical_font_size * 2)
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
            if patches[p]['chemical_0'] > patches[p]['chemical_1']:
                chem_type = "chemical_0"
            elif patches[p]['chemical_1'] > patches[p]['chemical_0']:
                chem_type = "chemical_1"
            else:
                idx = random.randint(0, 1)
                if idx == 0:
                    chem_type = "chemical_0"
                elif idx == 1:
                    chem_type = "chemical_1"

            chem = round(patches[p][chem_type]) * self.shade_strength
            pygame.draw.rect(
                self.screen,
                (0, chem if chem <= 255 else 255, 0) if chem_type == "chemical_0" else (chem if chem <= 255 else 255, chem if chem <= 255 else 255, 0),
                pygame.Rect(
                    p[0] - self.offset,
                    p[1] - self.offset,
                    self.patch_size,
                    self.patch_size
                )
            )

            if self.show_chem_text and (not sys.gettrace() is None or
                                        patches[p]['chemical_0'] >= self.sniff_threshold):  # if debugging show text everywhere, even 0
                text = self.chemical_font.render(str(round(patches[p]['chemical_0'], 1)), True, GREEN)
                self.screen.blit(text, text.get_rect(center=p))
            
            if self.show_chem_text and (not sys.gettrace() is None or
                                        patches[p]['chemical_1'] >= self.sniff_threshold):  # if debugging show text everywhere, even 0
                text = self.chemical_font.render(str(round(patches[p]['chemical_1'], 1)), True, YELLOW)
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
        "cluster_learners": 4,
        "scatter_learners": 4,
        "actions": [
            "random-walk",
            "drop-chemical-0",
            "drop-chemical-1",
            "move-toward-chemical-0",
            "move-toward-chemical-1",
            "move-away-chemical-0",
            "move-away-chemical-1"
        ],
        "sniff_threshold": 0.9,
        "sniff_patches": 5, 
        "diffuse_area": 0.5,
        "diffuse_radius": 0,
        "follow_mode": "det",
        #"follow_mode": "prob",
        "wiggle_patches": 5,
        "lay_area": 1,
        "lay_amount": 3,
        "evaporation": 0.90,
        "cluster_threshold": 1,
        "cluster_radius": 1,
        "obs_type": "paper",
        #"obs_type": "variation1",
        "normalize_rewards": False,
        "cluster_rew": 10,
        "cluster_penalty": -1,
        "scatter_rew": 0,
        "scatter_penalty": -1,
        "episode_ticks": 100,
        "W": 20,
        "H": 20,
        "PATCH_SIZE": 20,
        "TURTLE_SIZE": 16,
    }

    params_visualizer = {
      "FPS": 15,
      "SHADE_STRENGTH": 10,
      "SHOW_CHEM_TEXT": False,
      "CLUSTER_FONT_SIZE": 12,
      "CHEMICAL_FONT_SIZE": 8,
      "sniff_threshold": 0.9,
      "PATCH_SIZE": 20,
      "TURTLE_SIZE": 16,
      "show_dirs_view": False,
      "wiggle_patches": 3,
      "show_ph_view": False
    }

    from tqdm import tqdm

    EPISODES = 5
    SEED = 0
    np.random.seed(SEED)
    env = Slime(SEED, **params)
    env_vis = SlimeVisualizer(env.W_pixels, env.H_pixels, **params_visualizer)
    ACTION_NUM = len(params["actions"])
    AGENTS_NUM = env.cluster_learners + env.scatter_learners

    start_time = time.time()
    for ep in tqdm(range(1, EPISODES + 1), desc="Episode"):
        env.reset()
        for tick in tqdm(range(params['episode_ticks']), desc="Tick", leave=False):
            for agent in env.agent_iter(max_iter=AGENTS_NUM):
                observation, reward, _ , _, info = env.last(agent)
                #breakpoint()
                id = env.convert_observation(observation)
                action = np.random.randint(0, ACTION_NUM)
                env.step(action)
            env_vis.render(
                env.patches,
                env.learners,
                env.fov,
                env.ph_fov
            )
            #breakpoint()
        avg_cluster = env.avg_cluster()

    print("Total time = ", time.time() - start_time)
    env.close()

if __name__ == "__main__":
    main()