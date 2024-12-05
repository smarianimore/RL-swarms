import json
import random
import sys
from typing import Optional
from pprint import pprint
import time
import cProfile

import pygame

import gymnasium as gym
from gymnasium.spaces import Discrete, MultiBinary

import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import ObsType
from pettingzoo.test import api_test


class Ants(AECEnv):

    def observe(self, agent: str) -> ObsType:
        return np.array(self.observations[agent])

    def observation_space(self, agent):
        return self._observation_spaces[agent]
    
    def action_space(self, agent):
        return self._action_spaces[agent]
    
    metadata = {"render_modes": ["human", "server"]}


    def __init__(self,
                 seed,
                 render_mode: Optional[str] = None,
                 **kwargs):
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        np.random.seed(seed)
        random.seed(seed)
        
        self.population = kwargs['population']
        self.learner_population = kwargs['learner_population']
        self.sniff_threshold = kwargs['sniff_threshold']
        self.diffuse_area = kwargs['diffuse_area']
        self.smell_area = kwargs['smell_area']
        self.lay_area = kwargs['lay_area']
        self.lay_amount = kwargs['lay_amount']
        self.evaporation = kwargs['evaporation']
        self.diffuse_mode = kwargs['diffuse_mode']
        self.follow_mode = kwargs['follow_mode']
        self.cluster_threshold = kwargs['cluster_threshold']
        self.cluster_radius = kwargs['cluster_radius']
        self.reward = kwargs['rew']
        self.penalty = kwargs['penalty']
        self.episode_ticks = kwargs['episode_ticks']

        self.W = kwargs['W']
        self.H = kwargs['H']
        self.patch_size = kwargs['PATCH_SIZE']
        self.turtle_size = kwargs['TURTLE_SIZE']
        #self.fps = kwargs['FPS']
        #self.shade_strength = kwargs['SHADE_STRENGTH']
        #self.show_chem_text = kwargs['SHOW_CHEM_TEXT']
        #self.cluster_font_size = kwargs['CLUSTER_FONT_SIZE']
        #self.chemical_font_size = kwargs['CHEMICAL_FONT_SIZE']
        #self.gui = kwargs["gui"]

        self.coords = []
        self.offset = self.patch_size // 2
        self.W_pixels = self.W * self.patch_size
        self.H_pixels = self.H * self.patch_size
        for x in range(self.offset, (self.W_pixels - self.offset) + 1, self.patch_size):
            for y in range(self.offset, (self.H_pixels - self.offset) + 1, self.patch_size):
                self.coords.append((x, y))  # "centre" of the patch or turtle (also ID of the patch)

        pop_tot = self.population + self.learner_population
        self.possible_agents = [str(i) for i in range(self.population, pop_tot)]  # DOC learning agents IDs
        self._agent_selector = agent_selector(self.possible_agents)
        self.agent = self._agent_selector.reset()

        n_coords = len(self.coords)
        # create learners turtle
        #self.learners = {i: {"pos": self.coords[np.random.randint(n_coords)]} for i in range(self.population, pop_tot)}
        # create NON learner turtles

        self.nest_pos = self.coords[(len(self.coords) // 2) - (self.H // 2)]

        self.food_piles_pos = (
            (self.coords[0][0] + self.patch_size * int(self.W * 1/5), self.coords[0][1] + self.patch_size * int(self.H * 1/5)), 
            (self.coords[0][0] + self.patch_size * int(self.W * 1/6), self.coords[0][1] + self.patch_size * int(self.H * 3/4)), 
            (self.coords[n_coords - 1][0] - self.patch_size * int(self.W * 1/6), self.nest_pos[1])
        )
        #self.food_piles_rad = self.patch_size * 3
        
        self.patches_near_food = self._get_patches_near_food(self.food_piles_pos)

        self.patches_near_nest = self._get_patches_near_nest(self.nest_pos)

        self.learners = {i: 
            {"pos": self.nest_pos, "food": 0} 
            for i in range(self.population, pop_tot)
        }
        self.turtles = {i: 
            {"pos": self.nest_pos}
            for i in range(self.population)
        }

        # patches-own [chemical] - amount of pheromone in each patch
        self.patches = {
            self.coords[i]: {
                'id': i,
                #'nest': 0,
                #'food': 0,
                'chemical': 0.0,
                'turtles': []
            }
            for i in range(n_coords)
        }
        
        #for pos in self.patches_near_nest:
        #    self.patches[pos]['nest'] = 1
        #for key in self.patches_near_food.keys():
        #    pos = self.patches_near_food[key]
        #    for p in pos:
        #        self.patches[p]['food'] = 1 
        self.patches[self.nest_pos]["turtles"].extend([l for l in self.learners])
        self.patches[self.nest_pos]["turtles"].extend([l for l in self.turtles])
        
        # pre-compute relevant structures to speed-up computation during rendering steps
        # DOC {(x,y): [(x,y), ..., (x,y)]} pre-computed smell area for each patch, including itself
        self.smell_patches = self._find_neighbours(self.smell_area)
        # DOC {(x,y): [(x,y), ..., (x,y)]} pre-computed lay area for each patch, including itself
        self.lay_patches = self._find_neighbours(self.lay_area)
        # DOC {(x,y): [(x,y), ..., (x,y)]} pre-computed diffusion area for each patch, including itself
        if self.diffuse_mode == 'cascade':
            self.diffuse_patches = self._find_neighbours_cascade(self.diffuse_area)
        else:
            self.diffuse_patches = self._find_neighbours(self.diffuse_area)
        # DOC {(x,y): [(x,y), ..., (x,y)]} pre-computed cluster-check for each patch, including itself
        self.cluster_patches = self._find_neighbours(self.cluster_radius)

        self._action_spaces = {a: Discrete(5) for a in
                              self.possible_agents}  # DOC 0 = walk, 1 = lay_pheromone, 2 = follow_pheromone
        self._observation_spaces = {a: MultiBinary(2) for a in
                              self.possible_agents}  # DOC [0] = whether the turtle is in a cluster [1] = whether there is chemical in turtle patch

        #Different from AECEnv attribute self.rewards - only keeps last step rewards
        #self.rewards_cust = {i: [] for i in range(self.population, pop_tot)}
        #self.cluster_ticks = {i: 0 for i in range(self.population, pop_tot)}
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(self.population, pop_tot)))
        )
        
        #if self.gui:
        #    self.screen = pygame.display.set_mode((self.W_pixels, self.H_pixels))
        #    self.clock = pygame.time.Clock()
        #    pygame.font.init()
        #    self.cluster_font = pygame.font.SysFont("arial", self.cluster_font_size)
        #    self.chemical_font = pygame.font.SysFont("arial", self.chemical_font_size)
        #    self.first_gui = True

    """
    def _get_patches_near_food(self, food_pos):
        patches_near_foods = {}

        for i, fp in enumerate(food_pos):
            patches_near_foods[i] = [] 

            pos = (fp[0] - self.patch_size * 2, fp[0] + self.patch_size * 2)
            start = fp[1] - self.patch_size
            finish = fp[1] + self.patch_size * 2
            patches_near_foods[i].extend([
                (p, j) for p in pos for j in range(start, finish, self.patch_size)
            ]) 

            pos = (fp[0] - self.patch_size, fp[0], fp[0] + self.patch_size)
            start = fp[1] - self.patch_size * 2
            finish = fp[1] + self.patch_size * 3
            patches_near_foods[i].extend([
                (p, i) for p in pos for i in range(start, finish, self.patch_size)
            ])
            
        return patches_near_foods
    """
    
    def _get_patches_near_food(self, food_pos):
        patches_near_foods = []

        for i, fp in enumerate(food_pos):
            pos = (fp[0] - self.patch_size * 2, fp[0] + self.patch_size * 2)
            start = fp[1] - self.patch_size
            finish = fp[1] + self.patch_size * 2
            patches_near_foods.extend([
                (p, j) for p in pos for j in range(start, finish, self.patch_size)
            ]) 

            pos = (fp[0] - self.patch_size, fp[0], fp[0] + self.patch_size)
            start = fp[1] - self.patch_size * 2
            finish = fp[1] + self.patch_size * 3
            patches_near_foods.extend([
                (p, i) for p in pos for i in range(start, finish, self.patch_size)
            ])
            
        return patches_near_foods

    def _get_patches_near_nest(self, nest_pos):
        patches_near_nest = []

        for i in range(len(nest_pos)):
                pos = (nest_pos[0] - self.patch_size * 2, nest_pos[0] + self.patch_size * 2)
                start = nest_pos[1] - self.patch_size
                finish = nest_pos[1] + self.patch_size * 2
                
                patches_near_nest.extend([(p, j) for p in pos for j in range(start, finish, self.patch_size)]) 

                pos = (nest_pos[0] - self.patch_size, nest_pos[0], nest_pos[0] + self.patch_size)
                start = nest_pos[1] - self.patch_size * 2
                finish = nest_pos[1] + self.patch_size * 3
                patches_near_nest.extend([(p, i) for p in pos for i in range(start, finish, self.patch_size)])
                
        return tuple(patches_near_nest) 
        
    def _find_neighbours_cascade(self, area: int):
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
        return x % self.W_pixels, y % self.H_pixels

    # learners act
    def step(self, action: int):
        if(self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
            self._was_dead_step(action)
            return
        
        self.agent = self.agent_name_mapping[self.agent_selection]  # ID of agent

        observations, cluster_ticks, rewards_cust = self.process_agent(
            self.cluster_ticks,
            self.rewards_cust,
        )
        self.observations[str(self.agent)] = observations
        self.cluster_ticks = cluster_ticks
        self.rewards_cust = rewards_cust
        
        if action == 0:  # DOC walk
            patches, turtle = self.walk(
                self.patches,
                self.learners[self.agent]
            )
            self.patches = patches
            self.learners[self.agent] = turtle
        elif action == 1:  # DOC lay_pheromone
            self.patches = self.lay_pheromone(
                self.patches,
                self.learners[self.agent]['pos']
            )
        elif action == 2:  # DOC follow_pheromone
            max_pheromone, max_coords = self._find_max_pheromone(self.learners[self.agent]['pos'])
            
            if max_pheromone >= self.sniff_threshold:
                self.patches = self.follow_pheromone(
                    self.patches,
                    max_coords,
                    self.learners[self.agent]
                )
            else:
                patches, turtle = self.walk(
                    self.patches,
                    self.learners[self.agent]
                )
                self.patches = patches
                self.learners[self.agent] = turtle
        elif action == 3:
            # take food
            self.learners[self.agent] = self.take_food(self.learners[self.agent])
        elif action == 4:
            # drop food
            self.learners[self.agent] = self.drop_food(self.learners[self.agent])

        if self._agent_selector.is_last():
            for ag in self.agents:
                self.rewards[ag] = self.rewards_cust[self.agent_name_mapping[ag]][-1]
            if len(self.turtles) > 0:
                self.turtles, self.patches = self.move(self.turtles, self.patches)
            self.patches = self._evaporate(self.patches)
            self.patches = self._diffuse(self.patches)
            
            #if self.gui:
            #    self.render()
                #breakpoint()
        else:
            self._clear_rewards()
            
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[str(self.agent)] = 0
        self._accumulate_rewards()
        
    """
    def take_food(self, agent):
        for f in self.patches_near_food:
        if agent['pos'] in self.patches_near_food[f] and agent['food'] == 0:
            agent['food'] = 1
            self.patches_near_food[f].remove(agent['pos'])

        return agent
    """
    def take_food(self, agent):
        if agent['pos'] in self.patches_near_food and agent['food'] == 0:
            agent['food'] = 1
            self.patches_near_food.remove(agent['pos'])

        return agent

    def drop_food(self, agent):
        if agent['food'] == 1:
            agent['food'] = 0
            self.patches_near_food.append(agent['pos'])
        
        return agent

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

    def process_agent(self, cluster_ticks, rewards_cust):
        cluster = self._compute_cluster(self.agent)
        chemical = self._check_chemical(self.agent)
        observations = np.array([cluster >= self.cluster_threshold, chemical])
        cluster_ticks, rewards_cust, _ = self.reward_cluster_and_time_punish_time(
            cluster_ticks,
            rewards_cust,
            cluster
        )

        return observations, cluster_ticks, rewards_cust

    def lay_pheromone(self, patches, pos):
        for p in self.lay_patches[pos]:
            patches[p]['chemical'] += self.lay_amount
        
        return patches

    
    def _diffuse(self, patches):
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

    def _evaporate(self, patches):
        for patch in patches.keys():
            if patches[patch]['chemical'] > 0:
                patches[patch]['chemical'] *= self.evaporation

        return patches

    def walk(self, patches, turtle):      
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

    def follow_pheromone(self, patches, ph_coords, turtle):
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

    def _find_max_pheromone(self, pos):
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

    def _compute_cluster(self, current_agent):
        cluster = 1
        for p in self.cluster_patches[self.learners[current_agent]['pos']]:
            cluster += len(self.patches[p]['turtles'])

        return cluster

    def avg_cluster(self):
        cluster_sizes = []  # registra la dim. dei cluster
        for l in self.learners:
            cluster = []  # tiene conto di quali turtle sono in quel cluster
            for p in self.cluster_patches[self.learners[l]['pos']]:
                for t in self.patches[p]['turtles']:
                    cluster.append(t)
            cluster.sort()
            if cluster not in cluster_sizes:
                cluster_sizes.append(cluster)

        # cleaning process: confornta i cluster (nello stesso episodio) e se ne trova 2 con più del 90% di turtle uguali ne elimina 1
        for cluster in cluster_sizes:
            for cl in cluster_sizes:
                if cl != cluster:
                    intersection = list(set(cluster) & set(cl))
                    if len(intersection) > len(cluster) * 0.90:
                        cluster_sizes.remove(cl)

        # calcolo avg_cluster_size
        somma = 0
        for cluster in cluster_sizes:
            somma += len(cluster)
        avg_cluster_size = somma / len(cluster_sizes)

        return avg_cluster_size

    def _check_chemical(self, current_agent):
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
        self.agent = current_agent
        cluster = self._compute_cluster(self.agent)
        if cluster >= self.cluster_threshold:
            self.cluster_ticks[self.agent] += 1

        cur_reward = ((cluster ^ 2) / self.cluster_threshold) * self.reward + (
                ((self.episode_ticks - self.cluster_ticks[self.agent]) / self.episode_ticks) * self.penalty)

        self.rewards_cust[self.agent].append(cur_reward)
        return cur_reward

    def reward_cluster_and_time_punish_time(self, cluster_ticks, rewards_cust, cluster):  # DOC NetLogo rewardFunc8
        if cluster >= self.cluster_threshold:
            cluster_ticks[self.agent] += 1

        cur_reward = (cluster_ticks[self.agent] / self.episode_ticks) * self.reward + \
                     (cluster / self.cluster_threshold) * (self.reward ** 2) + \
                     (((self.episode_ticks - cluster_ticks[self.agent]) / self.episode_ticks) * self.penalty)

        rewards_cust[self.agent].append(cur_reward)
        return cluster_ticks, rewards_cust, cur_reward

    def reset(self, seed=None, return_info=True, options=None):
        # empty stuff
        pop_tot = self.population + self.learner_population
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
        self.observations = {a: np.full((2, ), False) for a in self.agents}
        
        self.patches_near_food = self._get_patches_near_food(self.food_piles_pos)
        # re-position learner turtle
        for l in self.learners:
            self.patches[self.learners[l]['pos']]['turtles'].remove(l)
            #self.learners[l]['pos'] = self.coords[np.random.randint(len(self.coords))]
            self.learners[l]['pos'] = self.nest_pos
            self.learners[l]['food'] = 0
            self.patches[self.learners[l]['pos']]['turtles'].append(l)  # DOC id of learner turtle
        # re-position NON learner turtles
        for t in self.turtles:
            self.patches[self.turtles[t]['pos']]['turtles'].remove(t)
            #self.turtles[t]['pos'] = self.coords[np.random.randint(len(self.coords))]
            self.turtles[t]['pos'] = self.nest_pos
            self.patches[self.turtles[t]['pos']]['turtles'].append(t)
        # patches-own [chemical] - amount of pheromone in the patch
        for p in self.patches:
            self.patches[p]['chemical'] = 0.0

        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

    def get_neighborood_chemical(self, agent, as_vectors=False):
        agent_pos = self.learners[agent]["pos"]
        smell_patches = self.smell_patches[agent_pos]
        
        output_mask = []
        for patch in smell_patches:
            output_mask.append(self.patches[patch]["chemical"] - self.patches[agent_pos]["chemical"]) if as_vectors else output_mask.append(self.patches[patch]["chemical"])

        return np.array([output_mask], dtype=np.float32)


BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
RED = (190, 0, 0)
ORANGE = (222, 89, 28)
GREEN = (0, 190, 0)
VIOLET = (151, 89, 154)
LIGHT_BLUE = (0,98,196),
#FOOD_COLORS = (
#    (0,68,137),
#    (0,98,196),
#    (0,127,255)
#)

class AntsVisualizer:
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

    def render(
        self,
        patches,
        learners,
        turtles,
        patches_near_nest,
        patches_near_food
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
                pygame.Rect(p[0] - self.offset,
                    p[1] - self.offset,
                    self.patch_size,
                    self.patch_size
                )
            )
            if self.show_chem_text and (not sys.gettrace() is None or
                                        self.patches[p][
                                            'chemical'] >= self.sniff_threshold):  # if debugging show text everywhere, even 0
                text = self.chemical_font.render(str(round(self.patches[p]['chemical'], 1)), True, GREEN)
                self.screen.blit(text, text.get_rect(center=p))

        # draw nest
        #pygame.draw.circle(self.screen, VIOLET, self.nest_pos, self.patch_size * 3) 
        
        for p in patches_near_nest:
            pygame.draw.rect(
                self.screen,
                VIOLET,
                pygame.Rect(p[0] - self.offset,
                    p[1] - self.offset,
                    self.patch_size,
                    self.patch_size
                )
            )

        # draw food piles
        #for pos, color in zip(self.food_piles_pos, FOOD_COLORS):
        #    pygame.draw.circle(self.screen, color, pos, self.food_piles_rad) 
        #    if self.food_piles_rad != 0:
        #        self.food_piles_rad -= self.turtle_size / 256 
        #num_vals = int(len(self.patches_near_food) / len(FOOD_COLORS)) 
        #for i in range(len(FOOD_COLORS)):
        #    for j in range(num_vals * i, num_vals * (i + 1)):
        #        pygame.draw.rect(
        #            self.screen,
        #            FOOD_COLORS[i],
        #            pygame.Rect(
        #                self.patches_near_food[j][0] - self.offset,
        #                self.patches_near_food[j][1] - self.offset,
        #                self.patch_size,
        #                self.patch_size
        #            )
        #        )

        for p in patches_near_food:
            pygame.draw.rect(
                self.screen,
                LIGHT_BLUE,
                pygame.Rect(p[0] - self.offset,
                    p[1] - self.offset,
                    self.patch_size,
                    self.patch_size
                )
            )

        counter = self._get_food_counts(patches_near_food)
        for i in counter:
            if counter[i] > 1:
                text = self.cluster_font.render(
                    str(counter[i]),
                    True,
                    WHITE
                )
                self.screen.blit(text, text.get_rect(center=i))

        # draw learners
        for learner in learners.values():
            if learner['food'] == 1:
                pygame.draw.circle(
                    self.screen,
                    ORANGE,
                    (learner['pos'][0], learner['pos'][1]),
                    self.turtle_size // 2
                )
            else:
                pygame.draw.circle(
                    self.screen,
                    RED,
                    (learner['pos'][0], learner['pos'][1]),
                    self.turtle_size // 2
                )
        # draw NON learners
        for turtle in turtles.values():
            pygame.draw.circle(self.screen, BLUE, (turtle['pos'][0], turtle['pos'][1]), self.turtle_size // 2)

        for p in patches:
            if len(patches[p]['turtles']) > 1:
                text = self.cluster_font.render(
                    str(len(patches[p]['turtles'])),
                    True,
                    RED if -1 in patches[p]['turtles'] else WHITE
                )
                self.screen.blit(text, text.get_rect(center=p))

        self.clock.tick(self.fps)
        pygame.display.flip()

        return pygame.surfarray.array3d(self.screen)

    def _get_food_counts(self, patches_near_food):
        counter = {}
        for f in patches_near_food:
            if f in counter.keys():
                counter[f] += 1
            else:
                counter[f] = 1
        return counter

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

def main():
    params = {
      "population": 0,
      "learner_population": 100,
      #"learner_population": 10,
      "sniff_threshold": 0.9,
      "diffuse_area": 3,
      "diffuse_mode": "cascade",
      "follow_mode": "prob",
      "smell_area": 3,
      "lay_area": 1,
      "lay_amount": 3,
      "evaporation": 0.9,
      "cluster_threshold": 30,
      "cluster_radius": 5,
      "rew": 100,
      "penalty": -1,
      "episode_ticks": 500,
      #"episode_ticks": 50,
      "W": 66,
      #"W": 40,
      "H": 38,
      #"H": 40,
      "PATCH_SIZE": 20,
      #"PATCH_SIZE": 10,
      "TURTLE_SIZE": 16,
      #"TURTLE_SIZE": 8,
    }

    params_visualizer = {
      "FPS": 30,
      "SHADE_STRENGTH": 10,
      "SHOW_CHEM_TEXT": False,
      "CLUSTER_FONT_SIZE": 12,
      "CHEMICAL_FONT_SIZE": 8,
      "gui": True,
      "sniff_threshold": 0.9,
      "PATCH_SIZE": 20,
      #"PATCH_SIZE": 10,
      "TURTLE_SIZE": 16,
      #"TURTLE_SIZE": 8,
    }

    from tqdm import tqdm

    EPISODES = 5
    LOG_EVERY = 1
    SEED = 0
    np.random.seed(SEED)
    env = Ants(SEED, **params)
    env_vis = AntsVisualizer(env.W_pixels, env.H_pixels, **params_visualizer)

    start_time = time.time()
    for ep in tqdm(range(1, EPISODES + 1), desc="Episode"):
        env.reset()
        for tick in tqdm(range(params['episode_ticks']), desc="Tick", leave=False):
            for agent in env.agent_iter(max_iter=params["learner_population"]):
                observation, reward, _ , _, info = env.last(agent)
                action = np.random.randint(0, 5)
                env.step(action)
            #breakpoint()
            env_vis.render(
                env.patches,
                env.learners,
                env.turtles,
                env.patches_near_nest,
                env.patches_near_food
            )

    print("Total time = ", time.time() - start_time)
    env.close()
    env_vis.close()

if __name__ == "__main__":
    #PARAMS_FILE = "multi-agent-env-params.json"
    main()
    #cProfile.run("main()", "possible_refactor_min.prof")
