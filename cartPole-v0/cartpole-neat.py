from neat import nn, population, statistics

import gym
import numpy as np
import os

MOVEMENT_RIGHT = 1
MOVEMENT_LEFT = 0 # Constants defined according to actions.

env = gym.make('CartPole-v0') #Creating the environment

def eval_fitness(genomes): 
	for genome in genomes:
		observation = env.reset()
		env.render()
		net = nn.create_feed_forward_phenotype(genome)
		fitness = 0
		while True:
			inputs = observation
			output = net.serial_activate(inputs) #compute the outputs
			if (output[0] >= 0):
				observation, reward, done, info = env.step(MOVEMENT_RIGHT)
			else:
				observation, reward, done, info = env.step(MOVEMENT_LEFT)

			fitness += reward

			env.render()

			if done:
				print fitness
				env.reset()
				break
				
		genome.fitness = fitness #assigning fitness to the genome.


pop = population.Population(os.getcwd() + '/cartPole-v0_configuration') # Attaching the config file required by the NEAT Library
pop.run(eval_fitness, 300)

best = pop.statistics.best_genome()

env.monitor.start('cartpole-experiment/', force=True)

streak = 0
best_phenotype = nn.create_feed_forward_phenotype(best)

observation = env.reset()
env.render()

while streak < 100:
	fitness = 0
	frames = 0
	while 1:
		inputs = observation

		# active neurons
		output = best_phenotype.serial_activate(inputs)
		if (output[0] >= 0):
			observation, reward, done, info = env.step(MOVEMENT_RIGHT)
		else:
			observation, reward, done, info = env.step(MOVEMENT_LEFT)

		fitness += reward

		env.render()
		frames += 1
		if frames >= 200:
			done = True
		if done:
			if fitness >= 195:
				print 'streak: %d' %streak
				streak += 1
			else:
				print fitness
				print 'streak: %d' % streak
				streak = 0
			env.reset()
			break
env.monitor.close()