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


class EnvEvaluator:
    def __init__(self, make_net, activate_net, max_env_steps=None, make_env=None,
                 env=None):
        if env is None:
            self.env = make_env()
        else:
            self.env = env
        self.make_net = make_net
        self.activate_net = activate_net
        self.max_env_steps = max_env_steps

    def eval_genome(self, genome, config, render=False):
        net = self.make_net(genome, config)

        fitness = 0
        state = self.env.reset()
        done = False

        step_num = 0
        while True:
            step_num += 1
            if self.max_env_steps is not None and step_num == self.max_env_steps:
                break

            actions = self.activate_net(net, state)
            if not done:
                if render:
                    self.env.render()
                state, reward, done, _ = self.env.step(actions)
                fitness += reward
            if done:
                break
        return fitness

