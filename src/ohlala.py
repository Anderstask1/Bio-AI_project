import visualize
import pickle
import os
import neat

# Save the winner.
config_path = os.path.join(os.path.dirname(__file__), "examples/adaptive/neat.cfg")
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path,
)
root_dir = "../best_genomes/"
stats_dir = "../best_genomes/stats/"

for file in os.listdir(root_dir):
    if os.path.isdir(os.path.join(root_dir, file)):
        continue
    with open(os.path.join(root_dir, file), 'rb') as f:
        genome = pickle.load(f)

    # visualize.plot_stats(genome, ylog=True, view=True, filename=(stats_dir + file + "-fitness.svg"))
    # visualize.plot_species(genome, view=True, filename=(stats_dir + file + "-speciation_1.svg"))

    node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
    visualize.draw_net(config, genome, True, node_names=node_names)

    visualize.draw_net(config, genome, view=True, node_names=node_names,
                       filename=(stats_dir + file + ".gv"))
    visualize.draw_net(config, genome, view=True, node_names=node_names,
                       filename=(stats_dir + file + "-enabled.gv"), show_disabled=False)
    visualize.draw_net(config, genome, view=True, node_names=node_names,
                       filename=(stats_dir + file + "-enabled-pruned.gv"), show_disabled=False,
                       prune_unused=True)
    break
