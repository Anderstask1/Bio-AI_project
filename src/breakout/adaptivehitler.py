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
from gym import wrappers
from pytorch_neat.env_eval import MultiEnvEvaluator
from pytorch_neat.activations import tanh_activation, relu_activation
from pytorch_neat.adaptive_linear_net import AdaptiveLinearNet

batch_size = 4
DEBUG = True

envs = [gym.make("BipedalWalker-v2")] * 4

# print(envs[0].observation_space.low)
# print(envs[0].action_space.low)
counter = 0
FOLDER_NAME = "best_genome_" + str(uuid.uuid4())


# envs[0] = gym.wrappers.Monitor(envs[0], "./vid", video_callable=lambda episode_id: episode_id % 1000 == 0, force=True)


def make_net(genome, config, _batch_size):
    input_coords = [(np.random.random(), np.random.random()) for x in range(1, 25)]
    output_coords = [(np.random.random(), np.random.random()) for _ in range(1, 5)]
    # input_coords = [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, -1.0]]
    # output_coords = [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]
    return AdaptiveLinearNet.create(
        genome,
        config,
        input_coords=input_coords,
        output_coords=output_coords,
        weight_threshold=0.4,
        batch_size=batch_size,
        activation=relu_activation,
        output_activation=relu_activation,
        device="cpu",
    )


def activate_net(net, states):
    outputs = net.activate(states).numpy()
    print(outputs)
    return outputs


evaluator = MultiEnvEvaluator(make_net, activate_net, envs=envs, batch_size=batch_size,
                              max_env_steps=1000)

pool = multiprocessing.Pool(processes=8)


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


@click.command()
@click.option("--n_generations", type=int, default=10000)
@click.option("--n_processes", type=int, default=1)
def run(n_generations, n_processes):
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

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)
    checkpointer = neat.Checkpointer(generation_interval=10, time_interval_seconds=None,
                                     filename_prefix="./checkpoints/neat-checkpoint-")
    pop.add_reporter(checkpointer)

    winner = pop.run(eval_genomes, n_generations)

    print(winner)
    final_performance = evaluator.eval_genome(winner, config)
    print("Final performance: {}".format(final_performance))
    generations = reporter.generation + 1
    return generations


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
