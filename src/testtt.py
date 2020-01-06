import neat
import gym
import numpy as np
from gym import wrappers
from neat import nn
from pytorch_neat.env_eval import EnvEvaluator
# from mountain_car.main import make_net, activate_net
from walker.main import make_net, activate_net, sqnl
# from breakout.main import make_net, activate_net
from neat.six_util import iteritems, itervalues

env = gym.make("BipedalWalker-v2")
# env = gym.wrappers.Monitor(env, "../../vid", video_callable=lambda episode_id: True, force=True)
# env = gym.wrappers.atari_preprocessing.AtariPreprocessing(env, frame_skip=1)
evaluator = EnvEvaluator(make_net, activate_net, env=env, max_env_steps=200)

pop = neat.Checkpointer.restore_checkpoint("../checkpoints/neat-checkpoint-BW-499")

genomes = list(iteritems(pop.population))
fitnesses = []
individual_num = 0
for genome in genomes:
    individual_num += 1
    fitness = evaluator.eval_genome(genome[1], pop.config)
    print("Individual number {0} with fitness {1:.2f}".format(individual_num, fitness))
    if not individual_num % 5:
        evaluator.eval_genome(genome[1], pop.config, render=True)
    fitnesses.append(fitness)
best_ind = np.argmax(fitnesses)
# print(best_ind)
best_fit = evaluator.eval_genome(genomes[best_ind][1], pop.config, render=True)
print(best_fit)
