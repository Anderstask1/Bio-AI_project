# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import os
import click
import neat
import gym
import visualize
from gym import wrappers
import numpy as np

env = gym.make("BipedalWalker-v2")


# envs[0] = gym.wrappers.Monitor(envs[0], "./vid", video_callable=lambda episode_id: episode_id % 1000 == 0, force=True)
# print(envs)

def sqnl(x):
    if x > 2:
        return 1
    elif x >= 0:
        return x - x ** 2 / 4
    elif x >= -2:
        return x + x ** 2 / 4
    else:
        return -1


def make_net(genome, config):
    return neat.nn.RecurrentNetwork.create(genome, config)


def activate_net(net, states):
    outputs = net.activate(states)
    # outputs = np.subtract(outputs, 1)
    return outputs


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = make_net(genome, config)

        fitness = 0
        state = env.reset()
        done = False

        while not done:
            action = activate_net(net, state)
            state, reward, done, _ = env.step(action)
            fitness += reward

        genome.fitness = fitness


def eval_genomes_parallel(genome, config):
    net = make_net(genome, config)

    fitness = 0
    state = env.reset()
    done = False

    while not done:
        action = activate_net(net, state)
        state, reward, done, _ = env.step(action)
        fitness += reward

    return fitness


@click.command()
@click.option("--n_generations", type=int, default=500)
@click.option("--n_processes", type=int, default=1)
@click.option("--n_save", type=int, default=100)
@click.option("--load_gen", type=str, default="")
def run(n_generations, n_processes, n_save, load_gen):
    config_path = os.path.join(os.path.dirname(__file__), "neat.cfg")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    # config.genome_config.add_activation("sqnl", sqnl)

    if load_gen:
        pop = neat.Checkpointer.restore_checkpoint("../checkpoints/neat-checkpoint-BW-" + load_gen)
    else:
        pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)
    checkpointer = neat.Checkpointer(generation_interval=n_save, time_interval_seconds=None,
                                     filename_prefix="../checkpoints/neat-checkpoint-BW-")
    pop.add_reporter(checkpointer)

    if n_processes > 1:
        pe = neat.ParallelEvaluator(n_processes, eval_genomes_parallel)
        winner = pop.run(pe.evaluate, n_generations)
    else:
        winner = pop.run(eval_genomes, n_generations)

    print(winner)
    generations = reporter.generation + 1
    stats.save_genome_fitness()

    visualize.draw_net(config, winner)
    visualize.plot_stats(stats, ylog=False)
    visualize.plot_species(stats)

    return generations


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
