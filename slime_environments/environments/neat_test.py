import os
import neat
import numpy as np
import pygame
import random
from typing import *
import time


PI = 3.14
EPSILON = 0.001


BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
RED = (190, 0, 0)
GREEN = (0, 190, 0)


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


# Reference:
# https://neat-python.readthedocs.io/en/latest/xor_example.html


EVAL_FITNESS_TICKS = 1000

ENV_GRID_W = 100
ENV_GRID_H = 100

PHEROMONE_EVAPORATION = 0.9
PHEROMONE_LAY_RADIUS = 1
PHEROMONE_LAY_AMOUNT = 255
PHEROMONE_SMELL_RADIUS = 3
PHEROMONE_DIFFUSE_FACTOR = 0.7

AGENT_SPEED = 0.7
AGENT_CLUSTER_RADIUS = 10

ACTION_RANDOM_WALK = 0
ACTION_LAY_PHEROMONE = 1
ACTION_FOLLOW_PHEROMONE = 2

WINDOW_H = 512
WINDOW_W = (WINDOW_H / ENV_GRID_H) * ENV_GRID_W
PATCH_W = WINDOW_W / ENV_GRID_W
PATCH_H = WINDOW_H / ENV_GRID_H

AGENT_TRIANGLE_H = 20.0
AGENT_TRIANGLE_W = AGENT_TRIANGLE_H * 0.9
AGENT_TRIANGLE_HALF_W = AGENT_TRIANGLE_W * 0.5
AGENT_TRIANGLE_Y_OFFSET = 0

config_file = os.path.join(SCRIPT_DIR, "neat_config")
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)

p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))
p.add_reporter(neat.StatisticsReporter())

# A dictionary holding:
# - id: matching NEAT's genome_id (starting from 1)
# - position
# - velocity
# - max_cluster_size: the biggest cluster this agent has been into
agents = {}

env_grid_agent_count = np.zeros((ENV_GRID_W, ENV_GRID_H), dtype=int)  # How many agents there are in each cell
env_grid_chemical = np.zeros((ENV_GRID_W, ENV_GRID_H), dtype=float)   # What's the chemical level in each cell

pygame.init()
pygame.display.set_caption("SLIME (Evolutionary Computation)")
pygame_screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
pygame_clock = pygame.time.Clock()
pygame.font.init()
pygame_font = pygame.font.SysFont("arial", 16)


def current_ms():
    return time.time() * 1000


def _agent_patch(agent) -> np.ndarray:
    return np.floor(agent['position']).astype(int)


def _agent_set_target(agent, target: np.ndarray) -> np.ndarray:
    dp = target - agent['position']
    agent['velocity'] = dp / np.linalg.norm(dp)


def reset():
    for agent_id in range(1, config.pop_size+1):
        agent = {
            'id': agent_id,
            'position': np.array([np.random.uniform(0.5, ENV_GRID_W - 0.5), np.random.uniform(0.5, ENV_GRID_H - 0.5)]),
            'velocity': np.random.random((2,)),
            'max_cluster_size': 1
        }
        agent['velocity'] /= np.linalg.norm(agent['velocity'])  # Normalize
        agents[agent_id] = agent

        env_grid_agent_count[*_agent_patch(agent)] += 1


def _rotate_2d_vector(vector: np.ndarray, theta: float) -> np.ndarray:
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta),  np.cos(theta)]])
    return np.dot(rotation_matrix, vector)


def _eval_cluster_size(agent) -> int:
    px, py = _agent_patch(agent)
    cluster_size = np.sum(env_grid_agent_count[
        max(px-AGENT_CLUSTER_RADIUS, 0):min(px+AGENT_CLUSTER_RADIUS+1, ENV_GRID_W),
        max(py-AGENT_CLUSTER_RADIUS, 0):min(py+AGENT_CLUSTER_RADIUS+1, ENV_GRID_H)
        ])
    return cluster_size


def observe(agent_id: int):
    """ Given an agent, returns its observation.
    
    This is the input to the agent's NN in order to determine the next action.
    """

    agent = agents[agent_id]

    velocity = agent['velocity']
    #hpheromone0, hpheromone1, hpheromone2 = _find_nearby_pheromone(agent_id, k=3)
    # Only see pheromone, doesn't see other agents

    observation = (
        *velocity,
        0,0,0,
        0,0,0,
        0,0,0
        # TODO DEBUG
        #*hpheromone0[0], hpheromone0[1],
        #*hpheromone1[0], hpheromone1[1],
        #*hpheromone2[0], hpheromone2[1]
        )
    return observation


def _find_nearby_pheromone(agent):
    result = []
    px, py = _agent_patch(agent)
    
    fx, tx = max(px-PHEROMONE_SMELL_RADIUS, 0), min(px+PHEROMONE_SMELL_RADIUS+1, ENV_GRID_W)
    fy, ty = max(py-PHEROMONE_SMELL_RADIUS, 0), min(py+PHEROMONE_SMELL_RADIUS+1, ENV_GRID_H)
    smell_area = env_grid_chemical[fx:tx, fy:ty]
    
    K = 3  # Maximum number of locations to return (ordered by intensity)
    k = min(K, smell_area.size)

    result = np.argpartition(smell_area.flatten(), -k)[-k:]

    def to_grid_coord(i: int):
        grid_coord = np.array(np.unravel_index(i, smell_area.shape)) + np.array([fx, fy])
        assert((grid_coord[0] >= 0 and grid_coord[0] < ENV_GRID_W) and (grid_coord[1] >= 0 and grid_coord[1] < ENV_GRID_H))
        return grid_coord

    result = [to_grid_coord(i) for i in result]
    return result


def _lay_pheromone(agent):
    px, py = _agent_patch(agent)
    env_grid_chemical[
        max(px-PHEROMONE_LAY_RADIUS, 0):min(px+PHEROMONE_LAY_RADIUS+1, ENV_GRID_W),
        max(py-PHEROMONE_LAY_RADIUS, 0):min(py+PHEROMONE_LAY_RADIUS+1, ENV_GRID_H),
        ] = PHEROMONE_LAY_AMOUNT


def perform_action(agent_id: int, agent_action: int) -> None:
    agent = agents[agent_id]

    if agent_action == ACTION_RANDOM_WALK:
        # Rotate the velocity vector by a random angle (very high probability not to rotate)
        MAX_ROTATION_ANGLE = np.pi / 6.0

        theta = pow(1.0 - random.random(), 12)
        theta = theta * (MAX_ROTATION_ANGLE * 2.0) - MAX_ROTATION_ANGLE

        agent['velocity'] = _rotate_2d_vector(agent['velocity'], theta)
    elif agent_action == ACTION_LAY_PHEROMONE:
        _lay_pheromone(agent)
    elif agent_action == ACTION_FOLLOW_PHEROMONE:
        hpheromone = _find_nearby_pheromone(agent)
        if len(hpheromone) == 0:
            return  # Stand still
        _agent_set_target(agent, hpheromone[0] + 0.5)
    else:
        raise Exception("Invalid action for agent %d: %d" % (agent_id, agent_action,))


def _diffuse_pheromone():
    global env_grid_chemical

    # Reference:
    # https://stackoverflow.com/questions/8102781/efficiently-doing-diffusion-on-a-2d-map-in-python

    new_array = np.copy(env_grid_chemical)

    # Right
    shifted_mask = PHEROMONE_DIFFUSE_FACTOR * np.roll(env_grid_chemical, shift=1, axis=1)
    shifted_mask[0, :] = 0 
    new_array = np.maximum(new_array, shifted_mask)

    # Left
    shifted_mask = PHEROMONE_DIFFUSE_FACTOR * np.roll(env_grid_chemical, shift=-1, axis=1)
    shifted_mask[-1, :] = 0 
    new_array = np.maximum(new_array, shifted_mask)

    # Down
    shifted_mask = PHEROMONE_DIFFUSE_FACTOR * np.roll(env_grid_chemical, shift=1, axis=0)
    shifted_mask[:, 0] = 0 
    new_array = np.maximum(new_array, shifted_mask)

    # Up
    shifted_mask = PHEROMONE_DIFFUSE_FACTOR * np.roll(env_grid_chemical, shift=-1, axis=0)
    shifted_mask[:, -1] = 0 
    new_array = np.maximum(new_array, shifted_mask)

    env_grid_chemical = new_array


def _evaporate_pheromone():
    global env_grid_chemical

    env_grid_chemical *= PHEROMONE_EVAPORATION


def render():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

    pygame_screen.fill(BLACK)

    env_to_win_s = np.array([WINDOW_W, WINDOW_H]) / np.array([ENV_GRID_W, ENV_GRID_H])

    # Render pheromone grid
    for ix in range(0, ENV_GRID_W):
        for iy in range(0, ENV_GRID_H):
            chemical = env_grid_chemical[ix, iy]
            pygame.draw.rect(pygame_screen, (0, min(chemical, 255), 0), pygame.Rect(ix * PATCH_W, iy * PATCH_H, PATCH_W, PATCH_H))

    # Render agents
    for agent_id, agent in agents.items():
        agent_pos = agent['position'] * env_to_win_s

        triangle_base_point = agent_pos + np.array([0, AGENT_TRIANGLE_Y_OFFSET])
        vertices = (
            triangle_base_point + agent['velocity'] * AGENT_TRIANGLE_H,
            triangle_base_point + np.array([agent['velocity'][1], -agent['velocity'][0]]) * AGENT_TRIANGLE_HALF_W,
            triangle_base_point - np.array([agent['velocity'][1], -agent['velocity'][0]]) * AGENT_TRIANGLE_HALF_W,
        )
        pygame.draw.polygon(pygame_screen, (255, 0, 0), vertices)

        #text = pygame_font.render(str(agent_id), True, (255, 255, 255))
        #pygame_screen.blit(text, text.get_rect(center=agent_pos))

    pygame.display.flip()


def _move_agent(agent):
    speed = AGENT_SPEED

    env_grid_agent_count[*_agent_patch(agent)] -= 1
    assert(env_grid_agent_count[*_agent_patch(agent)] >= 0)

    px, py = agent['position']
    vx, vy = agent['velocity']

    EPSILON = 0.1

    while speed > 0:
        ox = px
        oy = py
        px = ox + vx * speed
        py = oy + vy * speed
        step = speed

        if px < 0:
            step = (px - ox) / vx
            px = EPSILON
            vx = -vx
        elif px > ENV_GRID_W:
            step = (px - ox) / vx
            px = ENV_GRID_W - EPSILON
            vx = -vx

        if py < 0:
            step = (py - oy) / vy
            py = EPSILON
            vy = -vy
        elif py > ENV_GRID_H:
            step = (py - oy) / vy
            py = ENV_GRID_H - EPSILON
            vy = -vy
        
        speed -= step

    agent['position'][0] = px
    agent['position'][1] = py
    agent['velocity'][0] = vx
    agent['velocity'][1] = vy

    env_grid_agent_count[*_agent_patch(agent)] += 1


def update_agents(genomes, config):
    """ Function called by NEAT every generation.
    
    Every generation observes the agents' behavior for a fixed number of ticks and evaluates the fitness
    based on the maximum cluster size the agent has been in.
    """

    nn = {}

    # Create the NN for every agent based on the genome
    for agent_id, genome in genomes:
        nn[agent_id] = neat.nn.FeedForwardNetwork.create(genome, config)

    # Observe the agents' behavior for some time
    for tick in range(0, EVAL_FITNESS_TICKS):
        for agent_id, genome in genomes:
            agent = agents[agent_id]

            # Observe the current state and infer the action
            agent_state = observe(agent_id)
            
            action_prob_distr = nn[agent_id].activate(agent_state)
            agent_action = np.argmax(action_prob_distr)
            perform_action(agent_id, agent_action)

            _move_agent(agent)

            agent['max_cluster_size'] = max(_eval_cluster_size(agent), agent['max_cluster_size'])

        _diffuse_pheromone()
        _evaporate_pheromone()

        if tick % 100 == 0:
            print("Tick %d" % (tick,))

        render()

    # Update the genome fitness
    for agent_id, genome in genomes:
        agent = agents[agent_id]

        genome.fitness = agent['max_cluster_size']



def main():
    generation_i = 0
    while True:
        reset()

        best_agent = p.run(update_agents, n=1)

        generation_i += 1


if __name__ == "__main__":
    main()
