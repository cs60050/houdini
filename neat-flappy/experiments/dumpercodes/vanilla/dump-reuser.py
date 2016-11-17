# Author: Praveen Ch

import cPickle as pickle

from neat import nn, population, statistics

from ple.games.flappybird import FlappyBird
from ple import PLE

game = FlappyBird()

p = PLE(game, fps=30, display_screen=True, force_fps=False)
p.init()

def statify(observation):

	return list((observation['player_y'], observation['player_vel'], observation['next_pipe_dist_to_player'], observation['next_pipe_top_y'], observation['next_pipe_bottom_y']))

genome = pickle.load(open("genome_gen_0", "rb"))
p.init()
inputs = statify(game.getGameState())
net = nn.create_feed_forward_phenotype(genome)

print genome

fitness = 0
avg_fitness = 0
avg_score = 0

while True:
	fitness = 0
	while True:
		inputs = statify(game.getGameState())
		output = net.serial_activate(inputs) #compute the outputs
		
		if(output[0] > output[1]):
			reward = p.act(p.getActionSet()[0])
		else:
			reward = p.act(p.getActionSet()[1])

		sc = p.score()

		fitness = fitness + (reward+1)
		if p.game_over():
			avg_fitness += fitness
			avg_score += sc
			p.reset_game()
			break

print "Generation :"
