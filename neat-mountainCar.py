from __future__ import print_function
import os
import neat as neat
import numpy as np
import gym
from gym import wrappers
import visualize
import graphviz
import argparse
import uuid

def eval_genomes(genomes, config):
    steps=500
    render =False

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        #simulation 
        state = my_env.reset()
        cum_reward = 0.0

        for j in range(steps):
            #outputs = neat.nn.FeedForwardNetwork.activate(inputs)
            outputs = net.activate(state)
            action = np.argmax(outputs)
            state, reward, done, _ = my_env.step(action)
            #print("you got reward: ",reward)
            if render:
                my_env.render()
            if done:
                break

            cum_reward += reward

        genome.fitness = cum_reward


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # load checkpoint
    if args.checkpoint:
        p = neat.Checkpointer.restore_checkpoint(args.checkpoint)


    # Add a stdout reporter to show progress in the terminal.
    #p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(generation_interval=10, time_interval_seconds=None, filename_prefix='./checkpoints_mc/neat-checkpoint-'))


    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 2)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    #save and sort plots from simulation in folder
    unique_plot_filename =str(uuid.uuid4()) #make unique filename to not overwrite old plots
    fitness_plot_name= './plots_mc/fitness_'+str(winner.fitness)+'id_'+unique_plot_filename+'.svg'
    species_plot_name= './plots_mc/species_'+str(winner.fitness)+'id_'+unique_plot_filename+'.svg'
    network_plot_name= './plots_mc/network'+str(winner.fitness)+'id_'+unique_plot_filename+'.svg'

    visualize.draw_net(config, winner, view=True, filename=network_plot_name)
    visualize.plot_stats(stats, ylog=False, view=True, filename=fitness_plot_name)
    visualize.plot_species(stats, view=True, filename=species_plot_name)


def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='OpenAI Gym Solver')
    parser.add_argument('--checkpoint', type=str,
                    help="Uses a checkpoint to start the simulation")
    args = parser.parse_args()

    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    my_env = gym.make('MountainCar-v0')
    videos_dir = mkdir('.', 'videos')
    monitor_dir = mkdir(videos_dir, 'MountainCar-v0')

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward_car.txt')
    run(config_path)
