from ple.games.flappybird import FlappyBird
from ple import PLE
from collections import defaultdict
import numpy as np
import cPickle as pickle
import copy


class TabularQAgent():
	"""
	Agent implementing tabular Q-learning.
	"""

	def __init__(self, action_space):
		self.action_space = action_space
		self.action_n = len(action_space)
		self.config = {
			"init_mean" : 5,	  # Initialize Q values with this mean
			"init_std" : 0.2,	   # Initialize Q values with this standard deviation
			"learning_rate" : 0.2,
			"eps": 0.2,			# Epsilon in epsilon greedy policies
			"discount": 0.95,
			"n_iter": 10000		# Number of iterations
		}
		
		self.q = defaultdict(self.func)
		self.state = defaultdict(None)

	def func(self):
		return self.config["init_std"] * np.random.randn(self.action_n) + self.config["init_mean"]

	def pickAction(self, eps=None):
		if eps is None:
			eps = self.config["eps"]
			self.config["eps"] *= 0.99999
		# epsilon greedy with decaying epsilon.
		action = np.argmax(self.q[self.state]) if np.random.random() > eps else np.random.randint(0, self.action_n)
		return action
	def update(self, action, reward, observation, episode_over):
		if episode_over:
			future = -5
		else:
			future = np.max(self.q[observation])
		# print "Old Q value [", self.state, action, "] = ",  self.q[self.state][action]
		self.q[self.state][action] += self.config["learning_rate"] * (reward + self.config["discount"] * future - self.q[self.state][action])
		# print "New Q value [", self.state, action, "] = ",  self.q[self.state][action]
 		self.state = observation

game = FlappyBird()
p = PLE(game, fps=30, display_screen=True)
agent = TabularQAgent(action_space=p.getActionSet())
# print "action set = ", p.getActionSet()
p.init()
observation = game.getGameState()
observation = ((int(observation["player_y"]) - int(observation["next_pipe_bottom_y"])), int(observation["next_pipe_dist_to_player"]), int(observation["player_vel"]))
agent.state = observation
max_score = -10
episode_count = 0
output = open("out.txt", "w")
frame_count = 0
batch_sum = 0
# print "Initial State: ", observation
while True:
	frame_count += 1

	episode_over = False
	action = agent.pickAction()	
	# print "Action = ", action
	reward = p.act(p.getActionSet()[action])
	# print "Reward = ", reward
	if p.game_over():
		episode_over = True
		# print ">>>DEAD!"
	observation = game.getGameState()
	observation = ((int(observation["player_y"]) - int(observation["next_pipe_bottom_y"])), int(observation["next_pipe_dist_to_player"]), int(observation["player_vel"]))
	# print "Next observation = ", observation
	agent.update(action, reward, observation, episode_over)

	if episode_over:
		batch_sum += frame_count		
		episode_count += 1		
		if episode_count % 100 == 0:
			output.write("Episode " + str(episode_count) + ", Score = " + str(p.score()) + ", Avg Frames survived = " + str(batch_sum / 100) + "Q Size = " + str(len(agent.q)) + "\n")
			print "Episode ", episode_count, ", Score = ", p.score(), ", Avg Frames survived = ", batch_sum / 100, "Q Size = ", len(agent.q)
			batch_sum = 0
			if p.score > max_score:
				max_score = p.score
				# q_table = copy.deepcopy(agent.q)
				q_table = dict(agent.q)
				pickle.dump(q_table, open("agent_q.p", "w"))
		p.reset_game()
		observation = game.getGameState()
		observation = ((int(observation["player_y"]) - int(observation["next_pipe_bottom_y"])), int(observation["next_pipe_dist_to_player"]), int(observation["player_vel"]))
		agent.state = observation

		frame_count = 0
	# print "observation = ", observation
	# print "reward = ", reward
	# print game.getGameState(), "\n\n"

