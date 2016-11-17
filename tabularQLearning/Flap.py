# Author: Divyansh Gupta

from ple.games.flappybird import FlappyBird
from ple import PLE
from TabularQAgent import TabularQAgent

output_file = "out/out_4_full_new.txt"
output = open(output_file, "a")
agent_file = "pickle/agent_q_full_new.p"

training = True # False for testing

batch_size = 100
save_freq = 25000

def preprocess_reduce(observation):
	return ((int(observation["player_y"]) - int(observation["next_pipe_bottom_y"])) // 4, int(observation["next_pipe_dist_to_player"]) // 4, int(observation["player_vel"]) // 4)

def preprocess_discrete(observation):
	return ((int(observation["player_y"]) - int(observation["next_pipe_bottom_y"])), int(observation["next_pipe_dist_to_player"]), int(observation["player_vel"]))

preprocess = preprocess_discrete

game = FlappyBird()

#Pass force_fps=False, to see the game at 30fps
p = PLE(game, fps = 30, display_screen = True, force_fps=True)
if p.display_screen:
	p.init()

observation = game.getGameState()
state = preprocess(observation)

try:
	with open(agent_file, "r") as agent_dump:
		agent = TabularQAgent(action_space = p.getActionSet(), initial_state = state, agent_file = agent_dump)
	if training:
		print "WARNING: Saved agent already exists with this name. Continuing training. Make sure only one instance is training the same agent file."
		# training = False
		# p.display_screen = True
		# p.force_fps = False
except:
	agent = TabularQAgent(action_space = p.getActionSet(), initial_state = state)

episode_count = 0
batch_delta = 0
batch_frame = 0
batch_score = 0


# TODO: refactor to nested loop of episodes and frames
while True:
	batch_frame += 1

	action = agent.pickAction(training = training)

	reward = p.act(p.getActionSet()[action])
	observation = game.getGameState()
	state = preprocess(observation)
	done = True if p.game_over() else False
	
	if training:
		batch_delta += agent.updateQ(action, reward, next_state = state, done = done)
	else:
		agent.setState(state)

	if done:
		episode_count += 1
		batch_score += p.score()	

		if episode_count % batch_size == 0:
			avg_delta = batch_delta / batch_frame  	# Average change in Q value for batch_size episodes
			avg_frame = batch_frame / batch_size	# Average number of frames survived in batch_size episodes
			avg_score = batch_score / batch_size	# Average score in batch_size episodes

			out_line = "Episode: " + str(episode_count) + " AvgScore: " + str(avg_score) + " AvgFrames: " + str(avg_frame) + " QDelta: " + str(avg_delta) + " QSize: " + str(len(agent.q)) + "\n"
			output.write(out_line)
			print out_line
			
			batch_delta = 0
			batch_frame = 0
			batch_score = 0
		
		if training:
			if episode_count % save_freq == 0:
				with open(agent_file, 'w') as agent_dump:
					agent.saveAgent(agent_dump)
		p.reset_game()
		agent.setState(state)
