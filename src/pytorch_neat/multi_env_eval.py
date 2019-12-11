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

import numpy as np
import pickle
import uuid
import os


class MultiEnvEvaluator:
    def __init__(self, make_net, activate_net, save_individual=1000, batch_size=1, max_env_steps=None, make_env=None,
                 envs=None, visualize=False):
        if envs is None:
            self.envs = [make_env() for _ in range(batch_size)]
        else:
            self.envs = envs
        self.make_net = make_net
        self.activate_net = activate_net
        self.batch_size = batch_size
        self.max_env_steps = max_env_steps
        self.individual_index = 0
        self.last_genomes = []
        # if not visualize:
        #     self.folder_name = "best_genome_" + str(uuid.uuid4())
        #     os.makedirs(os.path.join("./best_genomes", self.folder_name), exist_ok=True)
        self.genome_save_counter = 0
        self.save_individual = save_individual

    def eval_genome(self, genome, config, render=False, debug=False, input_seeds=None):
        # self.individual_index += 1
        # if self.individual_index % self.save_individual == 0:
        #     self.save_best_genome(config)
        net = self.make_net(genome, config, self.batch_size)

        # seeds = []
        # for i, env in enumerate(self.envs):
        #     if input_seeds:
        #         seed = input_seeds[i]
        #     else:
        #         seed = np.random.randint(0, 1000000)
        #         seeds.append(seed)
        #     env.seed(seed)

        fitnesses = np.zeros(self.batch_size)
        states = [env.reset() for env in self.envs]
        dones = [False] * self.batch_size

        step_num = 0
        while True:
            step_num += 1
            if self.max_env_steps is not None and step_num == self.max_env_steps:
                break

            actions = self.activate_net(net, states)
            assert len(actions) == len(self.envs)
            for i, (env, action, done) in enumerate(zip(self.envs, actions, dones)):
                if not done:
                    if render:
                        env.render()
                        if step_num == 1:
                            env._elapsed_steps = 0
                    state, reward, done, _ = env.step(action)
                    fitnesses[i] += reward
                    if not done:
                        states[i] = state
                    dones[i] = done
            if all(dones):
                break
        avg_fitness = sum(fitnesses) / len(fitnesses)
        # self.last_genomes.append((genome, seeds))
        return avg_fitness

    # def save_best_genome(self):
    #     self.genome_save_counter += 1
    #     genomes_sorted_by_fitness = sorted(self.last_genomes, key=lambda kv: kv[1], reverse=True)
    #     best_genome = genomes_sorted_by_fitness[0]
    #     self.last_genomes.clear()
    #     with open(os.path.join("./best_genomes", self.folder_name, str(self.genome_save_counter)), 'wb+') as file:
    #         pickle.dump(best_genome, file)
    #     print("Genome with fitness {:.2f} saved".format(genomes_sorted_by_fitness[0][1]))
