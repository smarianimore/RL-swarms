import os
import neat
from neat.six_util import iterkeys
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

ENV_GRID_W = 256
ENV_GRID_H = 256

PHEROMONE_EVAPORATION = 0.94
PHEROMONE_LAY_RADIUS = 5
PHEROMONE_LAY_AMOUNT = 3
PHEROMONE_SMELL_RADIUS = 6
PHEROMONE_DIFFUSE_FACTOR = 0.98

AGENT_SPEED = 0.7
AGENT_CLUSTER_RADIUS = 10

ACTION_RANDOM_WALK = 0
ACTION_LAY_PHEROMONE = 1
ACTION_FOLLOW_PHEROMONE = 2

WINDOW_H = 512
WINDOW_W = (WINDOW_H / ENV_GRID_H) * ENV_GRID_W
PATCH_W = WINDOW_W / ENV_GRID_W
PATCH_H = WINDOW_H / ENV_GRID_H

AGENT_TRIANGLE_H = 14.0
AGENT_TRIANGLE_W = AGENT_TRIANGLE_H * 0.9
AGENT_TRIANGLE_HALF_W = AGENT_TRIANGLE_W * 0.5
AGENT_TRIANGLE_Y_OFFSET = 0

config_file = os.path.join(SCRIPT_DIR, "neat_config")
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)

def init_neat():
    """ Initialize the algorithm from scratch or load a checkpoint. """

    checkpoint_files = os.listdir('neat_checkpoints') if os.path.isdir('neat_checkpoints') else []
    if len(checkpoint_files) > 0:
        checkpoint_files.sort(reverse=True, key=lambda x: int(x.split('_')[1]))
        latest_checkpoint = os.path.join('neat_checkpoints', checkpoint_files[0])

        print(f"Restoring latest checkpoint: {latest_checkpoint}")
        return neat.Checkpointer.restore_checkpoint(latest_checkpoint)
    else:
        print(f"No checkpoint found")
        return neat.Population(config)

p = init_neat()

# A dictionary holding:
# - id: matching NEAT's genome_id (starting from 1)
# - position
# - velocity
# - max_cluster_size: the biggest cluster this agent has been into
agents = {}

env_grid_agent_count = np.zeros((ENV_GRID_W, ENV_GRID_H), dtype=int)  # How many agents there are in each patch
env_grid_chemical = np.zeros((ENV_GRID_W, ENV_GRID_H), dtype=float)   # What's the chemical level in each patch

pygame.init()
pygame.display.set_caption("SLIME (Evolutionary Computation)")
pygame_screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
pygame_clock = pygame.time.Clock()
pygame.font.init()
pygame_font = pygame.font.SysFont("arial", 16)


class Profiler:
    def __init__(self):
        self.stats = {}

    def _insert_dt(self, stat, dt: float):
        stat['min'] = min(stat['min'], dt)
        stat['max'] = max(stat['max'], dt)
        stat['sum'] += dt
        stat['count'] += 1

    def begin(self, key):
        def _calc_avg(stat):
            if stat['count'] == 0:
                return 0
            return stat['sum'] / stat['count']

        if key not in self.stats:
            self.stats[key] = {'st': 0, 'min': np.Infinity, 'max': 0, 'sum': 0, 'count': 0, 'avg': lambda: _calc_avg(self.stats[key])}
        self.stats[key]['st'] = time.time()

    def end(self, key):
        dt = time.time() - self.stats[key]['st']
        self._insert_dt(self.stats[key], dt)

    def reset(self, key=None):
        if key is None:
            self.stats = {}
        else:
            del self.stats[key]

    def __getitem__(self, key):
        avg_dt = self.stats[key]['avg']()
        if avg_dt >= 0.1:
            return f"{avg_dt:.1f} s"
        else:
            return f"{avg_dt * 1000:.3f} ms"

    def __str__(self):
        sorted_stats = list(self.stats.keys())  # Sort the stats by the worst to the best
        sorted_stats.sort(reverse=True, key=lambda k: self.stats[k]['avg']())

        return ', '.join([f"{k}: {self[k]}" for k in sorted_stats])


def _agent_patch(agent) -> np.ndarray:
    return np.floor(agent['position']).astype(int)


def _agent_set_target(agent, target: np.ndarray) -> np.ndarray:
    dp = target - agent['position']
    agent['velocity'] = dp / np.linalg.norm(dp)


def reset_env():
    global agents

    env_grid_agent_count[:, :] = 0
    env_grid_chemical[:, :] = 0

    agents = {}
    for agent_id in iterkeys(p.population):
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


def observe(agent):
    velocity = agent['velocity']

    hpheromones = _find_nearby_pheromone(agent)
    hpheromone0 = (hpheromones[0], env_grid_chemical[*hpheromones[0]],) if len(hpheromones) > 0 else ((0, 0), 0)
    hpheromone1 = (hpheromones[1], env_grid_chemical[*hpheromones[1]],) if len(hpheromones) > 1 else ((0, 0), 0)
    hpheromone2 = (hpheromones[2], env_grid_chemical[*hpheromones[2]],) if len(hpheromones) > 2 else ((0, 0), 0)

    observation = (
        *velocity,
        *hpheromone0[0], hpheromone0[1],
        *hpheromone1[0], hpheromone1[1],
        *hpheromone2[0], hpheromone2[1],
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

def _random_walk(agent):
    p = random.random()

    if p >= 0.1:
        return  # Very likely to keep the same velocity
    
    # Rotate the velocity vector by a random angle
    MAX_ROTATION_ANGLE = np.pi / 6.0
    theta = pow(1.0 - random.random(), 12)
    theta = theta * (MAX_ROTATION_ANGLE * 0.5) - MAX_ROTATION_ANGLE

    agent['velocity'] = _rotate_2d_vector(agent['velocity'], theta)


def _lay_pheromone(agent):
    px, py = _agent_patch(agent)
    lay_area = env_grid_chemical[
        max(px-PHEROMONE_LAY_RADIUS, 0):min(px+PHEROMONE_LAY_RADIUS+1, ENV_GRID_W),
        max(py-PHEROMONE_LAY_RADIUS, 0):min(py+PHEROMONE_LAY_RADIUS+1, ENV_GRID_H),
        ]
    lay_area += PHEROMONE_LAY_AMOUNT


def perform_action(agent_id: int, agent_action: int) -> None:
    agent = agents[agent_id]

    if agent_action == ACTION_RANDOM_WALK:
        _random_walk(agent)
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
    shifted_mask = PHEROMONE_DIFFUSE_FACTOR * np.roll(env_grid_chemical, shift=1, axis=0)
    shifted_mask[0, :] = 0 
    new_array = np.maximum(new_array, shifted_mask)

    # Left
    shifted_mask = PHEROMONE_DIFFUSE_FACTOR * np.roll(env_grid_chemical, shift=-1, axis=0)
    shifted_mask[-1, :] = 0 
    new_array = np.maximum(new_array, shifted_mask)

    # Down
    shifted_mask = PHEROMONE_DIFFUSE_FACTOR * np.roll(env_grid_chemical, shift=1, axis=1)
    shifted_mask[:, 0] = 0 
    new_array = np.maximum(new_array, shifted_mask)

    # Up
    shifted_mask = PHEROMONE_DIFFUSE_FACTOR * np.roll(env_grid_chemical, shift=-1, axis=1)
    shifted_mask[:, -1] = 0 
    new_array = np.maximum(new_array, shifted_mask)

    env_grid_chemical = new_array


def _evaporate_pheromone():
    global env_grid_chemical

    env_grid_chemical *= PHEROMONE_EVAPORATION


def _agent_color(agent):
    agent_id = agent['id']
    v = hash(agent_id)
    return (
        np.abs(np.sin(v * 0.4534)) * 128 + 30,
        np.abs(np.cos(v * 0.9879)) * 0,
        np.abs(np.cos(v * 0.1264)) * 128 + 30
    )

def render():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

    pygame_screen.fill(BLACK)

    env_to_win_s = np.array([WINDOW_W, WINDOW_H]) / np.array([ENV_GRID_W, ENV_GRID_H])

    # Render pheromone grid
    chemical_image = np.zeros((*env_grid_chemical.shape, 3,))
    chemical_image[:, :, 0] = np.where(env_grid_chemical[:, :] > 255, env_grid_chemical[:, :], 0)
    chemical_image[:, :, 1] = env_grid_chemical[:, :]
    chemical_image[:, :, 2] = np.where(env_grid_chemical[:, :] > 255, env_grid_chemical[:, :], 0)

    surface = pygame.surfarray.make_surface(chemical_image)
    surface = pygame.transform.scale(surface, (WINDOW_W, WINDOW_H))
    pygame_screen.blit(surface, dest=(0, 0))

    # Render agents
    for agent_id, agent in agents.items():
        agent_pos = agent['position'] * env_to_win_s

        p = agent_pos + np.array([0, AGENT_TRIANGLE_Y_OFFSET])
        vertices = (
            p + agent['velocity'] * AGENT_TRIANGLE_H,
            p + np.array([agent['velocity'][1], -agent['velocity'][0]]) * AGENT_TRIANGLE_HALF_W,
            p - np.array([agent['velocity'][1], -agent['velocity'][0]]) * AGENT_TRIANGLE_HALF_W,
        )

        pygame.draw.polygon(pygame_screen, _agent_color(agent), vertices)

        text = pygame_font.render(str(agent['max_cluster_size']), True, (255, 255, 255))
        pygame_screen.blit(text, text.get_rect(center=p))

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
    prof = Profiler()

    action_histogram = np.array([0, 0, 0])

    for tick in range(0, EVAL_FITNESS_TICKS):
        prof.begin('tick')

        for agent_id, genome in genomes:
            prof.begin('agent_update')

            agent = agents[agent_id]

            # Observe the current state and infer the action
            prof.begin('observe')
            agent_state = observe(agent)
            prof.end('observe')
            
            prof.begin('inference')
            action_prob_distr = nn[agent_id].activate(agent_state)
            agent_action = np.argmax(action_prob_distr)
            prof.end('inference')

            action_histogram[agent_action] += 1

            perform_action(agent_id, agent_action)

            _move_agent(agent)

            agent['max_cluster_size'] = max(_eval_cluster_size(agent), agent['max_cluster_size'])

            prof.end('agent_update')

        # Diffuse pheromone
        prof.begin('diffuse_pheromone')
        _diffuse_pheromone()
        prof.end('diffuse_pheromone')

        # Evaporate pheromone
        prof.begin('evaporate_pheromone')
        _evaporate_pheromone()
        prof.end('evaporate_pheromone')

        # Render
        prof.begin('render')
        render()
        prof.end('render')

        prof.end('tick')

        # 
        if tick % 100 == 0:
            print(f"Tick {tick:04d}; {prof}")
            prof.reset()

    action_histogram = action_histogram / np.sum(action_histogram)
    print(f"Action histogram; random_walk: {action_histogram[0]:.3f}, lay_pheromone: {action_histogram[1]:.3f}, follow_pheromone: {action_histogram[2]:.3f}")

    # Update the genome fitness
    for agent_id, genome in genomes:
        agent = agents[agent_id]
        genome.fitness = int(agent['max_cluster_size'])



def main():
    p.add_reporter(neat.StdOutReporter(True))

    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(generation_interval=5, filename_prefix='neat_checkpoints/checkpoint_')) 
    try:
        os.mkdir('neat_checkpoints')
    except:
        pass

    generation_i = 0
    while True:
        print("------------------------------------------------------------------------------------------------")
        print(f"Generation {generation_i}")
        print("------------------------------------------------------------------------------------------------")

        print("Population size: ", len(p.population))

        reset_env()

        best_agent = p.run(update_agents, n=1)

        generation_i += 1


if __name__ == "__main__":
    main()
