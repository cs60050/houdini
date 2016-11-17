# Author: Praveen Ch

from neat import nn, population, statistics

import os
import time

from ple.games.flappybird import FlappyBird
from ple import PLE

import cPickle as pickle

CLICK = 0
NO_CLICK = 1
NO_OF_RUNS = 20
RECORD_INTERVAL_GENERATIONS = 50
ALPHA = 0.1 #Decay const. Iterate over different values...


generation = 0

game = FlappyBird()

p = PLE(game, fps=30, display_screen=True)
p.init()

def statify(observation):

	return list((observation['player_y'], observation['player_vel'], observation['next_pipe_dist_to_player'], observation['next_pipe_top_y'], observation['next_pipe_bottom_y']))

def genome_dump(genome, generation):

	pickle.dump(genome, open("genome_gen_" + str(generation), "wb"))


def eval_fitness(genomes): 
	
	gen_summ_best_score = -1
	gen_summ_worst_score = 100000
	gen_summ_avg_score = 0

	gen_summ_best_fitness = -1
	gen_summ_worst_fitness = 100000
	gen_summ_avg_fitness = 0

	for genome in genomes:
		p.init()
		inputs = statify(game.getGameState())
		net = nn.create_feed_forward_phenotype(genome)
		fitness = 0
		avg_fitness = 0
		avg_score = 0

		for _ in range(NO_OF_RUNS):
			fitness = 0
			while True:
				inputs = statify(game.getGameState())
				output = net.serial_activate(inputs) #compute the outputs
				
				if(output[0] > output[1]):
					reward = p.act(p.getActionSet()[0])
				else:
					reward = p.act(p.getActionSet()[1])

				sc = p.getScore()

				fitness = fitness*(1-ALPHA) + (reward+1)*ALPHA
				if p.game_over():
					avg_fitness += fitness
					avg_score += sc
					p.reset_game()
					break

				# time.sleep(0.01)
		avg_fitness /= NO_OF_RUNS
		avg_score /= NO_OF_RUNS

		if gen_summ_best_fitness < avg_fitness:
			best_genome = genome
			gen_summ_best_fitness = avg_fitness
			gen_summ_best_score = avg_score

		if gen_summ_worst_fitness > avg_fitness:
			gen_summ_worst_fitness = avg_fitness
			gen_summ_worst_score = avg_score
		
		genome.fitness = avg_fitness

		gen_summ_avg_fitness += avg_fitness
		gen_summ_avg_score += avg_score

	if generation % RECORD_INTERVAL_GENERATIONS == 0:
		genome_dump(best_genome, generation)
	
	gen_summ_avg_fitness /= len(genomes)
	gen_summ_avg_score /= len(genomes)

	f = open("decay-dumper.stats", "a")
	f.write(str(gen_summ_best_score) + " " + str(gen_summ_worst_score) + " " + str(gen_summ_avg_score) + " " + str(gen_summ_best_fitness) + " " + str(gen_summ_worst_fitness) + " " + str(gen_summ_avg_fitness) )
	print str(gen_summ_best_score) + " " + str(gen_summ_worst_score) + " " + str(gen_summ_avg_score) + " " + str(gen_summ_best_fitness) + " " + str(gen_summ_worst_fitness) + " " + str(gen_summ_avg_fitness)
	generation += 1
	f.close()

pop = population.Population(os.getcwd() + '/vanilla-flappy.config')

pop.run(eval_fitness, 10000000)

