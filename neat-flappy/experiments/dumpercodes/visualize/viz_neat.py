# Author: Praveen Ch

# Script to visualize the network of NEAT genomes as images
# Requires graphviz

import cPickle as pickle
import visualize

for i in range(0, 500, 50):
	genome = pickle.load(open("genome_gen_" + str(i), "rb"))
	visualize.draw_net(genome, view=True, filename="genome_pruned_network"+str(i), show_disabled=False, prune_unused=True)
	visualize.draw_net(genome, view=True, filename="genome_network"+str(i))
	visualize.draw_net(genome, view=True, filename="genome_enabled_network"+str(i), show_disabled=False)
