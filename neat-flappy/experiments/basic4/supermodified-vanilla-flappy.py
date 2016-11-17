from neat import nn, population, statistics

import os
import time

from ple.games.flappybird import FlappyBird
from ple import PLE

CLICK = 0
NO_CLICK = 1
NO_OF_RUNS = 20


game = FlappyBird()

p = PLE(game, fps=30, display_screen=True)
p.init()

def statify(observation):

	return list((observation['player_y'], observation['player_vel'], observation['next_pipe_dist_to_player'], observation['next_pipe_top_y'], observation['next_pipe_bottom_y']))

def eval_fitness(genomes): 
	for genome in genomes:
		p.init()
		inputs = statify(game.getGameState())
		net = nn.create_feed_forward_phenotype(genome)
		fitness = 0
		avg_fitness = 0
	
		for _ in range(NO_OF_RUNS):
			fitness = 0
			while True:
				inputs = statify(game.getGameState())
				output = net.serial_activate(inputs) #compute the outputs
				
				if(output[0] > output[1]):
					reward = p.act(p.getActionSet()[0])
				else:
					reward = p.act(p.getActionSet()[1])

				fitness += reward + 1
				if p.game_over():
					avg_fitness += fitness
					p.reset_game()
					break

				# time.sleep(0.01)
		avg_fitness /= NO_OF_RUNS		
		genome.fitness = avg_fitness

pop = population.Population(os.getcwd() + '/supermodified-vanilla-flappy.config')

pop.run(eval_fitness, 10000000)

	