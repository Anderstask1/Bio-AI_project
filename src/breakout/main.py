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
from pytorch_neat.env_eval import EnvEvaluator
import numpy as np

env = gym.make("Breakout-v0")


# envs[0] = gym.wrappers.Monitor(envs[0], "./vid", video_callable=lambda episode_id: episode_id % 1000 == 0, force=True)
# print(envs)

env = gym.wrappers.atari_preprocessing.AtariPreprocessing(env, frame_skip=1)


def make_net(genome, config):
    return neat.nn.RecurrentNetwork.create(genome, config)


def activate_net(net, states):
    # print(states.flatten().shape)
    outputs = net.activate(states.ravel())
    # print(np.argmax(outputs))
    return np.argmax(outputs)


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
@click.option("--n_generations", type=int, default=100)
@click.option("--n_processes", type=int, default=8)
@click.option("--n_save", type=int, default=10)
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

    if load_gen:
        pop = neat.Checkpointer.restore_checkpoint("../checkpoints/neat-checkpoint-BO-" + load_gen)
    else:
        pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)
    checkpointer = neat.Checkpointer(generation_interval=n_save, time_interval_seconds=None,
                                     filename_prefix="../checkpoints/neat-checkpoint-BO-")
    pop.add_reporter(checkpointer)

    pe = neat.ParallelEvaluator(n_processes, eval_genomes_parallel)
    winner = pop.run(pe.evaluate, n_generations)

    print(winner)
    generations = reporter.generation + 1

    visualize.draw_net(config, winner, view=True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    return generations


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
