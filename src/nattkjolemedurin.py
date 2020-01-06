import pickle
import os
import neat
import re
import gym
from gym import wrappers
from pytorch_neat.env_eval import EnvEvaluator
from mountain_car.main import make_net, activate_net

env = gym.make("BipedalWalker-v2")
env = gym.wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: True, force=True)

config_path = os.path.join(os.path.dirname(__file__), "examples/adaptive/neat.cfg")
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path,
)
root_dir = "./best_genomes/genomes"
stats_dir = "stats/"

evaluator = EnvEvaluator(make_net, activate_net, env=env)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


for root, dirs, files in os.walk(root_dir, topdown=True):
    for file in sorted(files, key=natural_keys):
        if os.path.isdir(os.path.join(root_dir, file)):
            continue
        with open(os.path.join(root_dir, file), 'rb') as f:
            genome, og_fitness, seeds = pickle.load(f)
            print("Genomoe number", file)

        fitness = evaluator.eval_genome(genome, config)
        print("Original fitness", og_fitness)
        print("Current fitness", genome.fitness)
