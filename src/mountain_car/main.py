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
import numpy as np
import click
import neat
import gym
import uuid
import visualize
from gym import wrappers
from pytorch_neat.multi_env_eval import MultiEnvEvaluator
from pytorch_neat.recurrent_net import RecurrentNet

batch_size = 1
DEBUG = True

envs = [gym.make("MountainCar-v0")]

counter = 0
FOLDER_NAME = "best_genome_" + str(uuid.uuid4())


# envs[0] = gym.wrappers.Monitor(envs[0], "./vid", video_callable=lambda episode_id: episode_id % 1000 == 0, force=True)
# print(envs)


def make_net(genome, config, _batch_size):
    return RecurrentNet.create(genome, config, _batch_size)


def activate_net(net, states):
    outputs = net.activate(states).numpy()
    return np.argmax(outputs, axis=1)


@click.command()
@click.option("--n_generations", type=int, default=10000)
@click.option("--n_processes", type=int, default=1)
@click.option("--n_save", type=int, default=1000)
def run(n_generations, n_processes, n_save):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    config_path = os.path.join(os.path.dirname(__file__), "neat.cfg")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    evaluator = MultiEnvEvaluator(make_net, activate_net, save_individual=n_save, envs=envs, batch_size=batch_size,
                                  max_env_steps=200)

    if n_processes > 1:
        pool = multiprocessing.Pool(processes=n_processes)

        def eval_genomes(genomes, config):
            fitnesses = pool.starmap(
                evaluator.eval_genome, ((genome, config) for _, genome in genomes)
            )
            for (_, genome), fitness in zip(genomes, fitnesses):
                genome.fitness = fitness
            # best_genome = genomes[np.argmax(fitnesses)][1]
            # with open(os.path.join("./best_genomes", FOLDER_NAME), 'wb+') as file:
            #     pickle.dump(best_genome, file)
            # print("Genome with fitness {:.2f} saved".format(best_genome.fitness))
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
    checkpointer = neat.Checkpointer(generation_interval=10, time_interval_seconds=None,
                                     filename_prefix="../checkpoints/neat-checkpoint-MC-")
    pop.add_reporter(checkpointer)

    winner = pop.run(eval_genomes, 100)

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
