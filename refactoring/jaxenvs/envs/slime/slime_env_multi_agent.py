import sys
from typing import Optional
import numpy as np

import gymnasium as gym
from gymnasium.spaces import Discrete, MultiBinary

from pettingzoo import AECEnv
from pettingzoo.utils.env import ObsType

from jaxenvs.envs.world_env import WorldEnv

class Slime(AECEnv):
    metadata = {"render_modes": ["human", "server"]}

    def __init__(self,
                 render_mode: Optional[str] = None,
                 **kwargs):
        
        assert render_mode is None or render_mode is self.metadata["render_modes"]

        self.world = WorldEnv(**kwargs)

        # DOC 0 = walk, 1 = lay_pheromone, 2 = follow_pheromone
        self._action_spaces = {a: Discrete(3) for a in self.world.possible_agents}      

        # DOC [0] = whether the turtle is in a cluster [1] = whether there is chemical in turtle patch
        self._observation_spaces = {a: MultiBinary(2) for a in self.world.possible_agents}  
    
    def observe(self, agent: str) -> ObsType:
        return np.array(self.observations[agent])

    def observation_space(self, agent):
        return self._observation_spaces[agent]
    
    def action_space(self, agent):
        return self._action_spaces[agent]

    # learners act
    def step(self, action: int):
        if(self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
            self._was_dead_step(action)
            return
        
        agent_in_charge = self.world.agent_name_mapping[self.agent_selection]  # ID of agent
        
        self.process_agent(agent_in_charge) #
        self.state[agent_in_charge] = action #can ignore this
        
        if action == 0:  # DOC walk
            self.world.walk(self.world.learners[agent_in_charge], agent_in_charge)
        elif action == 1:  # DOC lay_pheromone
            self.world.lay_pheromone(self.world.learners[agent_in_charge]['pos'], self.world.lay_amount)
        elif action == 2:  # DOC follow_pheromone
            max_pheromone, max_coords = self.world._find_max_pheromone(self.world.learners[agent_in_charge]['pos'])
            if max_pheromone >= self.world.sniff_threshold:
                self.world.follow_pheromone(max_coords, self.world.learners[agent_in_charge], agent_in_charge)
            else:
                self.world.walk(self.world.learners[agent_in_charge], agent_in_charge)

        if self.world._agent_selector.is_last():
            for ag in self.agents:
                self.rewards[ag] = self.rewards_cust[self.world.agent_name_mapping[ag]][-1]
            # print("REW:",self.rewards)
            self.world.move()
            self.world._evaporate()
            self.world._diffuse()
            if self.world.gui:
                self.world.render()
        else:
            self._clear_rewards()
            
        self.agent_selection = self.world._agent_selector.next()
        # print(self.agent_selection)
        self._cumulative_rewards[str(agent_in_charge)] = 0
        # print("---")
        # print(self._cumulative_rewards)
        self._accumulate_rewards()
        # print(self._cumulative_rewards)
        # print("---")
        

    # not using ".change_all" method form BooleanSpace
    def process_agent(self, current_agent):
        #self._evaporate()
        #self._diffuse()

        self.agent = current_agent
        self.observations[str(self.agent)] = np.array([
            self.world._compute_cluster(self.agent) >=
            self.world.cluster_threshold, self.world._check_chemical(self.agent)
        ])
        self.reward_cluster_and_time_punish_time(self.agent)

    # not a real reward function
    def test_reward(self, current_agent):  # trying to invert rewards process, GOAL: check any strange behaviour
        """
        :return: the reward
        """
        self.agent = current_agent
        chem = 0
        for p in self.world.patches.values():
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

        :return: the reward
        """
        self.agent = current_agent
        cluster = self.world._compute_cluster(self.agent)
        if cluster >= self.world.cluster_threshold:
            self.cluster_ticks[self.agent] += 1

        cur_reward = ((cluster ^ 2) / self.world.cluster_threshold) * self.world.reward + (
            ((self.world.episode_ticks - self.cluster_ticks[self.agent]) / self.world.episode_ticks) * self.world.penalty)

        self.rewards_cust[self.agent].append(cur_reward)
        return cur_reward

    def reward_cluster_and_time_punish_time(self, current_agent):  # DOC NetLogo rewardFunc8
        """

        :return:
        """
        self.agent = current_agent
        cluster = self.world._compute_cluster(self.agent)
        if cluster >= self.world.cluster_threshold:
            self.cluster_ticks[self.agent] += 1

        cur_reward = (self.cluster_ticks[self.agent] / self.world.episode_ticks) * self.world.reward + \
                     (cluster / self.world.cluster_threshold) * (self.world.reward ** 2) + \
                     (((self.world.episode_ticks - self.cluster_ticks[self.agent]) / self.world.episode_ticks) * self.world.penalty)

        self.rewards_cust[self.agent].append(cur_reward)
        return cur_reward

    def reset(self, seed=None, return_info=True, options=None):
        # empty stuff
        pop_tot = self.world.population + self.world.learner_population
        self.rewards_cust = {i: [] for i in range(self.world.population, pop_tot)}
        self.cluster_ticks = {i: 0 for i in range(self.world.population, pop_tot)}
        
        #Initialize attributes for PettingZoo Env
        self.agents = self.world.possible_agents[:]
        self.world._agent_selector.reinit(self.agents)
        self.agent_selection = self.world._agent_selector.next()
        
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {a: np.full((2, ), False) for a in self.agents}
        
        # re-position learner turtle
        for l in self.world.learners:
            self.world.patches[self.world.learners[l]['pos']]['turtles'].remove(l)
            self.world.learners[l]['pos'] = self.world.coords[np.random.randint(len(self.world.coords))]
            self.world.patches[self.world.learners[l]['pos']]['turtles'].append(l)  # DOC id of learner turtle
        # re-position NON learner turtles
        for t in self.world.turtles:
            self.world.patches[self.world.turtles[t]['pos']]['turtles'].remove(t)
            self.world.turtles[t]['pos'] = self.world.coords[np.random.randint(len(self.world.coords))]
            self.world.patches[self.world.turtles[t]['pos']]['turtles'].append(t)
        # patches-own [chemical] - amount of pheromone in the patch
        for p in self.world.patches:
            self.world.patches[p]['chemical'] = 0.0

        self.world._agent_selector.reinit(self.agents)
        self.agent_selection = self.world._agent_selector.next()

