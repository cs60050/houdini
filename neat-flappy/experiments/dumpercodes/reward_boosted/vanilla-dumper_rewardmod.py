# Author: Praveen Ch

# Experiment with boosted reward

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

framefile = open("generation_frame", "w")
framefile.close()

gen_no = 0

game = FlappyBird()

p = PLE(game, fps=30, display_screen=True)
p.init()

def statify(observation):

	return list((observation['player_y'], observation['player_vel'], observation['next_pipe_dist_to_player'], observation['next_pipe_top_y'], observation['next_pipe_bottom_y']))

def genome_dump(genome, gen_no):

	pickle.dump(genome, open("genome_gen_" + str(gen_no), "wb"))


def processReward(raw_reward):
    raw_reward = round(raw_reward)
    if raw_reward == 0.0:
        return 15.0
    elif raw_reward == 1.0:
        return 10000.0
    elif raw_reward == -5.0:
        return -1000.0
    else:
    	return 0.0


def eval_fitness(genomes): 
	global gen_no

	gen_summ_best_score = -1
	gen_summ_worst_score = 100000
	gen_summ_avg_score = 0

	gen_summ_best_fitness = -100000
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
			frames = 0
			while True:
				frames += 1
				inputs = statify(game.getGameState())
				output = net.serial_activate(inputs) #compute the outputs
				
				if(output[0] > output[1]):
					reward = processReward(p.act(p.getActionSet()[0]))
				else:
					reward = processReward(p.act(p.getActionSet()[1]))

				sc = p.score()

				fitness = fitness + reward
				if p.game_over():
					with open("episode_frame", "a") as framefile:
						framefile.write(str(gen_no) + " " + str(frames) + "\n")
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

	if gen_no % RECORD_INTERVAL_GENERATIONS == 0:
		genome_dump(best_genome, gen_no)
	
	gen_summ_avg_fitness /= len(genomes)
	gen_summ_avg_score /= len(genomes)

	f = open("vanilla-dumper.stats", "a")
	f.write(str(gen_summ_best_score) + " " + str(gen_summ_worst_score) + " " + str(gen_summ_avg_score) + " " + str(gen_summ_best_fitness) + " " + str(gen_summ_worst_fitness) + " " + str(gen_summ_avg_fitness) + "\n")
	print str(gen_summ_best_score) + " " + str(gen_summ_worst_score) + " " + str(gen_summ_avg_score) + " " + str(gen_summ_best_fitness) + " " + str(gen_summ_worst_fitness) + " " + str(gen_summ_avg_fitness)
	gen_no += 1
	f.close()

pop = population.Population(os.getcwd() + '/vanilla-flappy.config')



pop.run(eval_fitness, 10000000)

