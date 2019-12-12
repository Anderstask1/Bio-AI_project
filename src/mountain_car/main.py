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

import multiprocessing
import os
import click
import neat
import gym
import visualize
from gym import wrappers
from pytorch_neat.env_eval import EnvEvaluator

env = gym.make("MountainCar-v0")

# env = gym.wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: episode_id % 1000 == 0, force=True)


def make_net(genome, config):
    return neat.nn.RecurrentNetwork.create(genome, config)


def activate_net(net, states):
    outputs = net.activate(states)
    return outputs


@click.command()
@click.option("--n_generations", type=int, default=10000)
@click.option("--n_processes", type=int, default=1)
@click.option("--n_save", type=int, default=100)
def run(n_generations, n_processes, n_save):
    config_path = os.path.join(os.path.dirname(__file__), "neat.cfg")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    evaluator = EnvEvaluator(make_net, activate_net, env=env, max_env_steps=200)

    if n_processes > 1:
        pool = multiprocessing.Pool(processes=n_processes)

        def eval_genomes(genomes, config):
            fitnesses = pool.starmap(
                evaluator.eval_genome, ((genome, config) for _, genome in genomes)
            )
            for (_, genome), fitness in zip(genomes, fitnesses):
                genome.fitness = fitness
    else:

        def eval_genomes(genomes, config):
            for i, (_, genome) in enumerate(genomes):
                try:
                    genome.fitness = evaluator.eval_genome(
                        genome, config)
                except Exception as e:
                    print(genome)
                    raise e

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)
    checkpointer = neat.Checkpointer(generation_interval=n_save, time_interval_seconds=None,
                                     filename_prefix="../checkpoints/neat-checkpoint-BW-")
    pop.add_reporter(checkpointer)

    winner = pop.run(eval_genomes, 10000)

    print(winner)
    final_performance = evaluator.eval_genome(winner, config, render=True)
    print("Final performance: {}".format(final_performance))
    generations = reporter.generation + 1

    visualize.draw_net(config, winner, view=True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    return generations


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
