
# Author: Divyansh Gupta

from collections import defaultdict
import numpy as np
import cPickle as pickle

class TabularQAgent():
	"""
	Agent implementing tabular Q-learning. 
	Tested with PLE Flappy Bird

	(Adapted from gym cartpole example)
	"""

	def __init__(self, action_space, initial_state, agent_file = None):
		self.action_space = action_space
		self.action_n = len(action_space)
		self.config = {
			"init_mean" : 5,	  # Initialize Q values with this mean
			"init_std" : 0.2,	   # Initialize Q values with this standard deviation
			"learning_rate" : 0.4,
			"eps": 0.5,			# Epsilon in epsilon greedy policies
			"eps_decay_rate": 0.9999,
			"discount": 0.95
		}
		
		if agent_file:
			q_table = pickle.load(agent_file)
			self.q = defaultdict(self.randQ, q_table)
		else:
			self.q = defaultdict(self.randQ)
		self.state = initial_state

	def randQ(self):
		return self.config["init_std"] * np.random.randn(self.action_n) + self.config["init_mean"]

	def pickAction(self, training = False):
		if not training or np.random.random() > self.config['eps']:
			action = np.argmax(self.q[self.state])
		else:
			action = np.random.randint(0, self.action_n)
			self.config["eps"] *= self.config["eps_decay_rate"]
		return action

	def updateQ(self, action, reward, next_state, done):
		if done:
			future = 0
		else:
			future = np.max(self.q[next_state])
		old_q = self.q[self.state][action]
		# print "Old Q value [", self.state, action, "] = ",  self.q[self.state][action]
		new_q = old_q + self.config["learning_rate"] * (reward + self.config["discount"] * future - old_q)
		# print "New Q value [", self.state, action, "] = ",  self.q[self.state][action]
 		self.q[self.state][action] = new_q
 		self.state = next_state

 		return abs(new_q - old_q)

 	def setState(self, state):
 		self.state = state

 	def saveAgent(self, file):
 		q_table = dict(self.q)
		pickle.dump(q_table, file)