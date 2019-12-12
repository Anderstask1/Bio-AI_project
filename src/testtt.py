import neat
import gym
import numpy as np
from gym import wrappers
from neat import nn
from pytorch_neat.env_eval import EnvEvaluator
# from mountain_car.main import make_net, activate_net
from walker.main import make_net, activate_net
from neat.six_util import iteritems, itervalues

env = gym.make("BipedalWalker-v2")
# envs[0] = gym.wrappers.Monitor(envs[0], "./vid", video_callable=lambda episode_id: True, force=True)
evaluator = EnvEvaluator(make_net, activate_net, env=env, max_env_steps=400)

pop = neat.Checkpointer.restore_checkpoint("../checkpoints/neat-checkpoint-BW-699-gc")

genomes = list(iteritems(pop.population))
fitnesses = []
individual_num = 0
for genome in genomes:
    individual_num += 1
    fitness = evaluator.eval_genome(genome[1], pop.config, render=True)
    print("Individual number {0} with fitness {1:.2f}".format(individual_num, fitness))
    fitnesses.append(fitness)
best_ind = np.argmax(fitnesses)
# print(best_ind)
best_fit = evaluator.eval_genome(genomes[best_ind][1], pop.config, render=True)
print(best_fit)
