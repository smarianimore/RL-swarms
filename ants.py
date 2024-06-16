"""
Reproduction of NetLogo Ants Model.
Date: 27/12/2023
WHAT IS IT?
a colony of ants forages for food. Though each ant follows a set of simple rules, the colony as a whole acts in a sophisticated way.
HOW IT WORKS?
When an ant finds a piece of food, it carries the food back to the nest, dropping a chemical as it moves. When other ants “sniff” the chemical, they follow the chemical toward the food. As more ants carry food to the nest, they reinforce the chemical trail.
"""

import json
import random
import sys
from typing import Optional

import gymnasium as gym
from gymnasium.spaces import Discrete, MultiBinary

import numpy as np
import pygame
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import ObsType
from pettingzoo.test import api_test


BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
RED = (190, 0, 0)
GREEN = (0, 190, 0)
PURPLE = (128, 0, 128)
SKY = (135, 206, 235)
ORANGE = (255, 164, 32)

class Ants(AECEnv):

    def observe(self, agent: str) -> ObsType:
        return np.array(self.observations[agent])

    def observation_space(self, agent):
        return self._observation_spaces[agent]
    
    def action_space(self, agent):
        return self._action_spaces[agent]
    
    metadata = {"render_modes": ["human", "server"]}

    
    def __init__(self,
                 render_mode: Optional[str] = None,
                 **kwargs):
        super().__init__()
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
        self.fps = kwargs['FPS']
        self.shade_strength = kwargs['SHADE_STRENGTH']
        self.show_chem_text = kwargs['SHOW_CHEM_TEXT']
        self.cluster_font_size = kwargs['CLUSTER_FONT_SIZE']
        self.chemical_font_size = kwargs['CHEMICAL_FONT_SIZE']
        self.gui = kwargs["gui"]

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

        # create learners turtle (RED = not carrying food)
        self.learners = {i: {"pos": self.coords[int((int(self.W/2)+1/2)*self.H)],"color":RED} for i in range(self.population, pop_tot)}
        # create NON learner turtles
        self.turtles = {i: {"pos": self.coords[int((int(self.W/2)+1/2)*self.H)],"color":RED} for i in range(self.population)}
        
        # patches-own [chemical] - amount of pheromone in each patch
        self.patches = {self.coords[i]: {"id": i,
                                         'chemical': 0.0,
                                         'food': 0, #amount of food on this patch(1 or 2)
                                         'turtles': [],
                                         'nest': 0, # 0=false, 1=true(nest-patches)
                                         'food_source_number': 0,
                                        'nest_scent': 0 #number (1, 2, or 3) to identify the food sources
                                         } for i in range(n_coords)}
        
        #setup nest and food sources
        center_x = self.coords[int((int(self.W/2)+1/2)*self.H)][0]
        center_y = self.coords[int((int(self.W/2)+1/2)*self.H)][1]

        for c in self.coords:
            x = c[0]
            y = c[1]
            #nest
            distance = self.distance(x, y,center_x,center_y)
            if distance < 50:    
                self.patches[c]['chemical'] = 0.0
                self.patches[c]['nest'] = 1
            if self.distance(x, y, self.coords[int(5 * self.W)][0], self.coords[int(5 * self.W)][1]) < 50:
                self.patches[c]['food_source_number'] = 1
            elif self.distance(x, y,  self.coords[int(10 * self.W)][0], self.coords[int(10 * self.W)][1]) < 50:
                self.patches[c]['food_source_number'] = 2
            elif self.distance(x, y, self.coords[int(25 * self.W)][0], self.coords[int(25 * self.W)][1]) < 50:
                self.patches[c]['food_source_number'] = 3
            if self.patches[c]['food_source_number'] > 0:
                self.patches[c]['food'] = random.choice([1, 2])
            
        for l in self.learners:
            self.patches[self.learners[l]['pos']]['turtles'].append(l)  # DOC id of learner turtles
        for t in self.turtles:
            self.patches[self.turtles[t]['pos']]['turtles'].append(t)

        # pre-compute relevant structures to speed-up computation during rendering steps
        # DOC {(x,y): [(x,y), ..., (x,y)]} pre-computed smell area for each patch, including itself
        self.smell_patches = {}
        self._find_neighbours(self.smell_patches, self.smell_area)
        # DOC {(x,y): [(x,y), ..., (x,y)]} pre-computed lay area for each patch, including itself
        self.lay_patches = {}
        self._find_neighbours(self.lay_patches, self.lay_area)
        # DOC {(x,y): [(x,y), ..., (x,y)]} pre-computed diffusion area for each patch, including itself
        self.diffuse_patches = {}
        if self.diffuse_mode == 'cascade':
            self._find_neighbours_cascade(self.diffuse_patches, self.diffuse_area)
        else:
            self._find_neighbours(self.diffuse_patches, self.diffuse_area)
        # DOC {(x,y): [(x,y), ..., (x,y)]} pre-computed cluster-check for each patch, including itself
        self.cluster_patches = {}
        self._find_neighbours(self.cluster_patches, self.cluster_radius)

        self._action_spaces = {a: Discrete(4) for a in
                              self.possible_agents} # DOC 0 = walk, 1 = follow_pheromone --> only red ants (without food)
                                                    # DOC 2 = go to the nest, 3 = go to the nest + lay_pheromone --> only orange ants (with food)
        self.observations = {a: MultiBinary(2) for a in
                              self.possible_agents}  # DOC [0] = whether the turtle is in a cluster [1] = whether there is chemical in turtle patch

        if self.gui:
            self.screen = pygame.display.set_mode((self.W_pixels, self.H_pixels))
            self.clock = pygame.time.Clock()
            pygame.font.init()
            self.cluster_font = pygame.font.SysFont("arial", self.cluster_font_size)
            self.chemical_font = pygame.font.SysFont("arial", self.chemical_font_size)
            self.first_gui = True

        #Complete rewards tracking for each agent
        #Different from AECEnv attribute self.rewards - only keeps last step rewards
        self.rewards_cust = {i: [] for i in range(self.population, pop_tot)}
        self.cluster_ticks = {i: 0 for i in range(self.population, pop_tot)}
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(self.population, pop_tot)))
        )

        #distance from food of the past tick{ id_agent: [distanza_dal_cumulo_1,distanza_dal_cumulo_2, distanza_dal_cumulo_3]}
        self.food_distances = {a: self.get_food_distances(a) for a in self.learners}

    def distance(self, x1, y1, x2, y2):
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def get_food_distances(self, agent):
        agent_pos = self.learners[agent]["pos"]
        distances = [np.inf, np.inf, np.inf]  # Inizializza le distanze con infinito

        for p in self.patches:
            if self.patches[p]['food_source_number'] > 0:
                food_source_number = self.patches[p]['food_source_number']
                distance = self.distance(agent_pos[0], agent_pos[1], p[0], p[1])
                if distance < distances[food_source_number - 1]:
                    distances[food_source_number - 1] = distance

        return distances

    def _find_neighbours_cascade(self, neighbours: dict, area: int):
        """
        For each patch, find neighbouring patches within square radius 'area', 1 step at a time
        (visiting first 1-hop patches, then 2-hops patches, and so on)

        :param neighbours: empty dictionary to fill
            (will be dict mapping each patch to list of neighouring patches {(x, y): [(nx, ny), ...], ...})
        :param area: integer representing the number of patches to consider in the 8 directions around each patch
        :return: None (1st argument modified as side effect)
        """
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
            # neighbours[p] = list(set(neighbours[p]))

    def _find_neighbours(self, neighbours: dict, area: int):
        """
        For each patch, find neighbouring patches within square radius 'area'

        :param neighbours: empty dictionary to fill
            (will be dict mapping each patch to list of neighouring patches {(x, y): [(nx, ny), ...], ...})
        :param area: integer representing the number of patches to consider in the 8 directions around each patch
        :return: None (1st argument modified as side effect)
        """
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

    def _wrap(self, x: int, y: int):
        """
        Wrap x,y coordinates around the torus

        :param x: the x coordinate to wrap
        :param y: the y coordinate to wrap
        :return: the wrapped x, y
        """
        if x < 0:
            x = self.W_pixels + x
        elif x > self.W_pixels:
            x = x - self.W_pixels
        if y < 0:
            y = self.H_pixels + y
        elif y > self.H_pixels:
            y = y - self.H_pixels
        return x, y

    # learners act
    def step(self, action: int):
        if(self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
            self._was_dead_step(action)
            return
        
        agent_in_charge = self.agent_name_mapping[self.agent_selection]  # ID of agent

        """print(f"\nMy name is {agent_in_charge}")
        
        if self.learners[agent_in_charge]['color'] == RED:
            print("I'm a red ant")
        elif self.learners[agent_in_charge]['color'] == ORANGE:
            print("I'm an orange ant")
        else:
            print("ERROR: I'm an ant, but I am neither red nor orange")

        self.state[agent_in_charge] = action #can ignore this

        print(f"I was told to execute action {action}")"""

        if self.learners[agent_in_charge]['color'] == RED:
            if action == 0:  # DOC walk
                self.walk(self.learners[agent_in_charge], agent_in_charge)
            elif action == 1:  # DOC follow_pheromone
                max_pheromone, max_coords = self._find_max_pheromone(self.learners[agent_in_charge]['pos'])
                if max_pheromone >= self.sniff_threshold:
                    self.follow_pheromone(max_coords, self.learners[agent_in_charge], agent_in_charge)
                else:
                    self.walk(self.learners[agent_in_charge], agent_in_charge)

            pos = self.learners[agent_in_charge]['pos']
            if self.patches[pos]['food'] > 0:
                self.learners[agent_in_charge]['color'] = ORANGE
                self.patches[pos]['food'] -= 1

        elif self.learners[agent_in_charge]['color'] == ORANGE:
            if action == 2:  # DOC go to the nest
                max_pheromone, max_coords = self.uphill_nest_scent(self.learners[agent_in_charge]['pos'])
                self.follow_pheromone(max_coords, self.learners[agent_in_charge], agent_in_charge)
            elif action == 3:  # DOC go to the nest + lay_pheromone
                max_pheromone, max_coords = self.uphill_nest_scent(self.learners[agent_in_charge]['pos'])
                self.follow_pheromone(max_coords, self.learners[agent_in_charge], agent_in_charge)
                self.lay_pheromone(self.learners[agent_in_charge]['pos'], self.lay_amount)

            #se siamo arrivati al nido 
            if self.patches[self.learners[agent_in_charge]['pos']]['nest'] == 1:
                self.learners[agent_in_charge]['color'] = RED
        
        # compute rewards
        self.process_agent(agent_in_charge) #
        self.food_distances[agent_in_charge] = self.get_food_distances(agent_in_charge) 


        if self._agent_selector.is_last():
            for ag in self.agents:
                self.rewards[ag] = self.rewards_cust[self.agent_name_mapping[ag]][-1]
            # print("REW:",self.rewards)
            self.move()
            self._evaporate()
            self._diffuse()
            self.render()
        else:
            self._clear_rewards()

        self.agent_selection = self._agent_selector.next()
        # print(self.agent_selection)
        self._cumulative_rewards[str(agent_in_charge)] = 0
        # print("---")
        # print(self._cumulative_rewards)
        self._accumulate_rewards()
        # print(self._cumulative_rewards)
        # print("---")
        
    # non learners act
    def move(self):
        for turtle in self.turtles:
            pos = self.turtles[turtle]['pos']
            t = self.turtles[turtle]
            if self.turtles[turtle]['color'] == RED: #not carrying food? look for it
                if self.patches[pos]['food'] > 0:
                    self.turtles[turtle]['color'] = ORANGE
                    self.patches[pos]['food'] -= 1
                    self.lay_pheromone(self.turtles[turtle]['pos'], self.lay_amount)
                    max_pheromone, max_coords = self.uphill_nest_scent(pos)
                    self.follow_pheromone(max_coords, t, turtle)
                else: 
                    max_pheromone, max_coords = self._find_max_pheromone(pos)
                    if max_pheromone >= self.sniff_threshold:
                        self.follow_pheromone(max_coords, t, turtle)
                    else:
                        self.walk(t, turtle)

            elif self.turtles[turtle]['color'] == ORANGE: #carrying food? take it back to the nest
                self.lay_pheromone(self.turtles[turtle]['pos'], self.lay_amount)
                max_pheromone, max_coords = self.uphill_nest_scent(pos)
                self.follow_pheromone(max_coords, t, turtle)
                if self.patches[self.turtles[turtle]['pos']]['nest'] == 1:
                    self.turtles[turtle]['color'] = RED

            
    
    def uphill_nest_scent(self,pos):
        max_ph = -1
        max_pos = [pos]
        for p in self.smell_patches[pos]:
            chem = self.patches[p]['nest_scent']
            if chem > max_ph:
                max_ph = chem
                max_pos = [p]
            elif chem == max_ph:
                max_pos.append(p)
        winner = max_pos[np.random.choice(len(max_pos))]

        return max_ph, winner

    def _is_carrying_food(self, current_agent):
        """
        Checks if an agent is carrying food (in other word, its color is orange)

        :return a boolean
        """
        if self.learners[current_agent]['color'] == ORANGE:
            food = True 
        else:
            food = False

        return food 

    def _check_chemical_around(self, current_agent):
        """
        Checks if in the sorrounding patches there is pheromone.

        :return a boolean
        """
        found = False
        
        for p in self.smell_patches[self.learners[current_agent]['pos']]:
            chem = self.patches[p]['chemical']
            if chem > self.sniff_threshold:
                found = True    
        
        return found

    # not using ".change_all" method form BooleanSpace
    def process_agent(self, current_agent):
        #self._evaporate()
        #self._diffuse()

        self.agent = current_agent
        food = self._is_carrying_food(current_agent)
        phero = self._check_chemical_around(current_agent)
        self.observations[str(self.agent)] = np.array([food , phero])

        #print(f"observations: food:{food}, phero: {phero}")

        intermediate_reward = 0
        if self.learners[current_agent]['color'] == RED:
            intermediate_reward = self.calculate_intermediate_reward(current_agent)

        total_reward = intermediate_reward + self.reward_cluster_punish_time(self.agent)
            
        #print(f"Total reward of {total_reward}")

        self.rewards_cust[self.agent].append(total_reward)
        return total_reward
        

    def lay_pheromone(self, pos, amount: int):
        """
        Lay 'amount' pheromone in square 'area' centred in 'pos'
        :param pos: the x,y position taken as centre of pheromone deposit area
        :param amount: the amount of pheromone to deposit
        :return: None (environment properties are changed as side effect)
        """
        for p in self.lay_patches[pos]:
            self.patches[p]['chemical'] += amount

    def _diffuse(self):
        """
        Diffuses pheromone from each patch to nearby patches controlled through self.diffuse_area patches in a way
        controlled through self.diffuse_mode:
            'simple' = Python-dependant (dict keys "ordering")
            'rng' = random visiting
            'sorted' = diffuse first the patches with more pheromone
            'filter' = do not re-diffuse patches receiving pheromone due to diffusion

        :return: None (environment properties are changed as side effect)
        """
        n_size = len(self.diffuse_patches[list(self.patches.keys())[0]])  # same for every patch
        patch_keys = list(self.patches.keys())
        if self.diffuse_mode == 'rng':
            random.shuffle(patch_keys)
        elif self.diffuse_mode == 'sorted':
            patch_list = list(self.patches.items())
            patch_list = sorted(patch_list, key=lambda t: t[1]['chemical'], reverse=True)
            patch_keys = [t[0] for t in patch_list]
        elif self.diffuse_mode == 'filter':
            patch_keys = [k for k in self.patches if self.patches[k]['chemical'] > 0]
        elif self.diffuse_mode == 'rng-filter':
            patch_keys = [k for k in self.patches if self.patches[k]['chemical'] > 0]
            random.shuffle(patch_keys)
        for patch in patch_keys:
            p = self.patches[patch]['chemical']
            ratio = p / n_size
            if p > 0:
                diffuse_keys = self.diffuse_patches[patch][:]
                for n in diffuse_keys:
                    self.patches[n]['chemical'] += ratio
                self.patches[patch]['chemical'] = ratio

    def _evaporate(self):
        """
        Evaporates pheromone from each patch according to param self.evaporation

        :return: None (environment properties are changed as side effect)
        """
        for patch in self.patches.keys():
            if self.patches[patch]['chemical'] > 0:
                self.patches[patch]['chemical'] *= self.evaporation

    def walk(self, turtle, _id: int):
        """
        Action 0: move in random direction (8 sorrounding cells)

        :param _id: the id of the turtle to move
        :param turtle: the turtle to move (dict mapping 'pos' to position as x,y)
        :return: None (pos is updated after movement as side-effect)
        """
        choice = [self.patch_size, -self.patch_size, 0]
        x, y = turtle['pos']
        self.patches[turtle['pos']]['turtles'].remove(_id)
        x2, y2 = x + np.random.choice(choice), y + np.random.choice(choice)
        x2, y2 = self._wrap(x2, y2)
        turtle['pos'] = (x2, y2)
        self.patches[turtle['pos']]['turtles'].append(_id)

    def follow_pheromone(self, ph_coords, turtle, _id: int):
        """
        Action 2: move turtle towards greatest pheromone found
        :param _id: the id of the turtle to move
        :param ph_coords: the position where max pheromone has been sensed
        :param turtle: the turtle looking for pheromone
        :return: None (pos is updated after movement as side-effect)
        """
        x, y = turtle['pos']
        self.patches[turtle['pos']]['turtles'].remove(_id)
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
        self.patches[turtle['pos']]['turtles'].append(_id)

    def _find_max_pheromone(self, pos):
        """
        Find where the maximum pheromone level is within a square controlled by self.smell_area centred in 'pos'.
        Following pheromone modeis controlled by param self.follow_mode:
            'det' = follow greatest pheromone
            'prob' = follow greatest pheromone probabilistically (pheromone strength as weight)

        :param pos: the x,y position of the turtle looking for pheromone
        :return: the maximum pheromone level found and its x,y position
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

    def _compute_cluster(self, current_agent):
        """
        Checks whether the learner turtle is within a cluster, given 'cluster_radius' and 'cluster_threshold'

        :return: a boolean
        """
        self.agent = current_agent
        cluster = 1
        for p in self.cluster_patches[self.learners[self.agent]['pos']]:
            cluster += len(self.patches[p]['turtles'])

        return cluster

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
        
        # re-position learner turtle
        for l in self.learners:
            self.patches[self.learners[l]['pos']]['turtles'].remove(l)
            self.learners[l]['pos'] = self.coords[int((int(self.W/2)+1/2)*self.H)]
            self.learners[l]['color'] = RED
            self.patches[self.learners[l]['pos']]['turtles'].append(l)  # DOC id of learner turtle
        # re-position NON learner turtles
        for t in self.turtles:
            self.patches[self.turtles[t]['pos']]['turtles'].remove(t)
            self.turtles[t]['pos'] = self.coords[int((int(self.W/2)+1/2)*self.H)]
            self.turtles[t]['color'] = RED
            self.patches[self.turtles[t]['pos']]['turtles'].append(t)

        # patches-own [chemical] - amount of pheromone in the patch
        center_x = self.coords[int((int(self.W/2)+1/2)*self.H)][0]
        center_y = self.coords[int((int(self.W/2)+1/2)*self.H)][1]
        for c in self.coords:
            x = c[0]
            y = c[1]
            #nest
            distance = self.distance(x, y,center_x,center_y)
            self.patches[c]['nest_scent'] = 10000 - distance
            if distance < 50:    
                self.patches[c]['chemical'] = 0.0
                self.patches[c]['nest'] = 1
            if self.distance(x, y, self.coords[int(5 * self.W)][0], self.coords[int(5 * self.W)][1]) < 50:
                self.patches[c]['food_source_number'] = 1
            elif self.distance(x, y,  self.coords[int(10 * self.W)][0], self.coords[int(10 * self.W)][1]) < 50:
                self.patches[c]['food_source_number'] = 2
            elif self.distance(x, y, self.coords[int(25 * self.W)][0], self.coords[int(25 * self.W)][1]) < 50:
                self.patches[c]['food_source_number'] = 3
            if self.patches[c]['food_source_number'] > 0:
                self.patches[c]['food'] = random.choice([1, 2])
            
            
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.food_distances = {a: self.get_food_distances(a) for a in self.learners}

   
    def close(self):
        if self.gui:
            if self.screen is not None:
                pygame.display.quit()
                pygame.quit()
                
                
    def get_neighborood_chemical(self, agent, as_vectors=False):
        agent_pos = self.learners[agent]["pos"]
        smell_patches = self.smell_patches[agent_pos]
        
        output_mask = []
        for patch in smell_patches:
            output_mask.append(self.patches[patch]["chemical"] - self.patches[agent_pos]["chemical"]) if as_vectors else output_mask.append(self.patches[patch]["chemical"])

        return np.array([output_mask], dtype=np.float32)

    def render(self, **kwargs):
        if self.gui:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # window closed -> program quits
                    pygame.quit()

            if self.first_gui:
                self.first_gui = False
                pygame.init()
                pygame.display.set_caption("ANTS")

            self.screen.fill(BLACK)
    
            # draw food source  and draw nest
            for p in self.patches:      
                if self.patches[p]['nest'] == 1:
                    pygame.draw.rect(self.screen, PURPLE,
                                 pygame.Rect(p[0] - self.offset, p[1] - self.offset, self.patch_size, self.patch_size))
                elif self.patches[p]['food'] > 0:
                    if self.patches[p]['food_source_number'] == 1:
                        #cyan
                        pygame.draw.rect(self.screen, pygame.Color("cyan"),
                                 pygame.Rect(p[0] - self.offset, p[1] - self.offset, self.patch_size, self.patch_size))
                    elif self.patches[p]['food_source_number'] == 2:
                        #sky
                        pygame.draw.rect(self.screen, SKY,
                                 pygame.Rect(p[0] - self.offset, p[1] - self.offset, self.patch_size, self.patch_size))
                    elif self.patches[p]['food_source_number'] == 3:
                        #blue
                        pygame.draw.rect(self.screen, BLUE,
                                 pygame.Rect(p[0] - self.offset, p[1] - self.offset, self.patch_size, self.patch_size))
                elif self.patches[p]['nest'] == 0:
                    chem = round(self.patches[p]['chemical']) * self.shade_strength
                    pygame.draw.rect(self.screen, (0, chem if chem <= 255 else 255, 0),
                                    pygame.Rect(p[0] - self.offset, p[1] - self.offset, self.patch_size, self.patch_size))
                       
            # draw learners
            for learner in self.learners.values():
                pygame.draw.circle(self.screen, learner['color'], (learner['pos'][0], learner['pos'][1]), self.turtle_size // 2)
            # draw NON learners
            for turtle in self.turtles.values():
                pygame.draw.circle(self.screen, turtle['color'], (turtle['pos'][0], turtle['pos'][1]), self.turtle_size // 2)
            
            self.clock.tick(self.fps)
            pygame.display.flip()
            return pygame.surfarray.array3d(self.screen)
        
    def _find_neighbours_cascade(self, neighbours: dict, area: int):
        """
        For each patch, find neighbouring patches within square radius 'area', 1 step at a time
        (visiting first 1-hop patches, then 2-hops patches, and so on)

        :param neighbours: empty dictionary to fill
            (will be dict mapping each patch to list of neighouring patches {(x, y): [(nx, ny), ...], ...})
        :param area: integer representing the number of patches to consider in the 8 directions around each patch
        :return: None (1st argument modified as side effect)
        """
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
            # neighbours[p] = list(set(neighbours[p]))

    def _wrap(self, x: int, y: int):
        """
        Wrap x,y coordinates around the torus

        :param x: the x coordinate to wrap
        :param y: the y coordinate to wrap
        :return: the wrapped x, y
        """
        if x < 0:
            x = self.W_pixels + x
        elif x > self.W_pixels:
            x = x - self.W_pixels
        if y < 0:
            y = self.H_pixels + y
        elif y > self.H_pixels:
            y = y - self.H_pixels
        return x, y
    
    def _compute_cluster(self, current_agent):
        """
        Checks whether the learner turtle is within a cluster, given 'cluster_radius' and 'cluster_threshold'

        :return: a boolean
        """
        self.agent = current_agent
        cluster = 1
        for p in self.cluster_patches[self.learners[self.agent]['pos']]:
            cluster += len(self.patches[p]['turtles'])

        return cluster
        
    def _check_chemical(self, current_agent):
        """
        Checks whether there is pheromone on the patch where the learner turtle is

        :return: a boolean
        """
        self.agent = current_agent
        return self.patches[self.learners[self.agent]['pos']][
                   'chemical'] > self.sniff_threshold
    
    def reward_cluster_punish_time(self, current_agent):  # DOC NetLogo rewardFunc7
        """
        Reward is (positve) if the agent finds food or return to the nest.

        :return: the reward
        """
        self.agent = current_agent
        if self.patches[self.learners[current_agent]['pos']]['food'] > 1:
            #print("Got standard reward!")
            cur_reward = self.reward
        else:
            cur_reward = 0

        self.rewards_cust[self.agent].append(cur_reward)
        return cur_reward
    
    def calculate_intermediate_reward(self, agent):
        current_min = np.min(self.get_food_distances(agent))
        #print(f"current distances: {current_min}")
        past_min = np.min(self.food_distances[agent])
        #print(f"past distances: {past_min}") 
        reward = 0

        if current_min < past_min:
            #print("Got intermediate reward!")
            reward = 10  # Un piccolo reward per avvicinarsi al cibo
        else:
            #print("Got negative reward!")
            reward = -1

        return reward

    def get_neighborood_chemical(self, agent, as_vectors=False):
        agent_pos = self.learners[agent]["pos"]
        smell_patches = self.smell_patches[agent_pos]
        
        output_mask = []
        for patch in smell_patches:
            output_mask.append(self.patches[patch]["chemical"] - self.patches[agent_pos]["chemical"]) if as_vectors else output_mask.append(self.patches[patch]["chemical"])

        return np.array([output_mask], dtype=np.float32)
    
    def avg_cluster(self):
        """
        Record the cluster size
        :return: avg cluster size
        """
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

    def _check_food(self):
        """
        Check if there is still food

        :return: boolean
        """

        for p in self.patches:      
            if self.patches[p]['food'] > 0:
                if (self.patches[p]['food_source_number'] == 1 or self.patches[p]['food_source_number'] == 2 or self.patches[p]['food_source_number'] == 3) and self.patches[p]['food'] > 0 :
                    return True #almeno un patch ha ancora il cibo



if __name__ == '__main__':
    PARAMS_FILE = "env-ants-params.json"
    EPISODES = 5
    LOG_EVERY = 1

    with open(PARAMS_FILE) as f:
        params = json.load(f)
        
    if params["gui"]:
        render = "human"
    else:
        render = "server"
        
    env = Ants(render_mode=render, **params)

    # Reset the environment to start the simulation
    env.reset()

    for ep in range(1, EPISODES + 1):
        env.reset()
        print(f"-------------------------------------------\nEPISODE: {ep}\n-------------------------------------------")
        for tick in range(params['episode_ticks']):
            #for agent in env.agent_iter(max_iter=params["learner_population"]):
                #observation, reward, _ , _, info = env.last(agent)
                #env.step(env.action_space(agent).sample())
            
            #env.evaporate_chemical()
            env.move()
            env._evaporate()
            env._diffuse()
            env.render()
            
    # End of simulation
    print("Simulation completed.")

    env.close()
