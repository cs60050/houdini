"""
Solve PLE Flappy Bird with DQN.

TensorFlow code for gym-cartpole-v0 by TiehHung Chuang (imironhead), 2016-09-06
tf-slim code by Jeremy Karnowski jkarnows, 2016-09-07
Adapted for Flappy Bird by Divyansh Gupta, 2016-11-13

Note: Parameter values Tabular Q Implementation

This verion does not include a separate Target Q Network copy
"""

from __future__ import absolute_import
# from __future__ import print_function
from __future__ import division
import numpy as np
import random
import gym
import tensorflow as tf
import tensorflow.contrib.slim as slim

from ple.games.flappybird import FlappyBird
from ple import PLE

version = 7
PARAM_K = 1

class DeepQLearningAgent(object):

    def __init__(self, dim_state, action_space, network_layers):
        self._action_space = action_space
        self._dim_state = dim_state
        self._dim_action = len(action_space)
        self._batch_size = 64
        self._gamma = 0.85

        self._prev_state = None
        self._prev_action = None

        prev_states = tf.placeholder(tf.float32, [None, self._dim_state])
        net = slim.stack(prev_states, slim.fully_connected, network_layers, activation_fn=tf.nn.relu, scope='fc')
        prev_action_values = slim.fully_connected(net, self._dim_action, activation_fn=None, scope='qvalues')
        prev_action_masks = tf.placeholder(tf.float32, [None, self._dim_action])
        prev_values = tf.reduce_sum(tf.mul(prev_action_values, prev_action_masks), reduction_indices=1)

        prev_rewards = tf.placeholder(tf.float32, [None, ])
        next_states = tf.placeholder(tf.float32, [None, self._dim_state])
        net = slim.stack(next_states, slim.fully_connected, network_layers, activation_fn=tf.nn.relu, scope='fc', reuse=True)
        next_action_values = slim.fully_connected(net, self._dim_action, activation_fn=None, scope='qvalues', reuse=True)
        next_values = prev_rewards + self._gamma * tf.reduce_max(next_action_values, reduction_indices=1)

        loss = tf.reduce_mean(tf.square(prev_values - next_values))
        training = tf.train.RMSPropOptimizer(learning_rate=2.5e-4, decay=0.9, momentum=0.95, epsilon=0.01).minimize(loss)

        self._tf_action_value_predict = prev_action_values
        self._tf_prev_states = prev_states
        self._tf_prev_action_masks = prev_action_masks
        self._tf_prev_rewards = prev_rewards
        self._tf_next_states = next_states
        self._tf_training = training
        self._tf_loss = loss
        self._tf_session = tf.InteractiveSession()

        self._tf_session.run(tf.initialize_all_variables())

        # Build the D which keeps experiences.
        self._time = 0
        self._epsilon = 0.2
        self._epsilon_min = 1e-5
        # self._epsilon_decay_time = 1000000
        self._epsilon_decay_rate = 0.999999
        self._experiences_max = 100000
        self._experiences_min = 50000
        self._experiences_num = 0
        self._experiences_prev_states = np.zeros((self._experiences_max, self._dim_state))
        self._experiences_next_states = np.zeros((self._experiences_max, self._dim_state))
        self._experiences_rewards = np.zeros((self._experiences_max))
        self._experiences_actions_mask = np.zeros((self._experiences_max, self._dim_action))

    def create_experience(self, prev_state, prev_action, reward, next_state):
        """
        keep an experience for later training.
        """
        if self._experiences_num >= self._experiences_max:
            idx = np.random.choice(self._experiences_max)
        else:
            idx = self._experiences_num

        self._experiences_num += 1

        self._experiences_prev_states[idx] = np.array(prev_state)
        self._experiences_next_states[idx] = np.array(next_state)
        self._experiences_rewards[idx] = reward
        self._experiences_actions_mask[idx] = np.zeros(self._dim_action)
        self._experiences_actions_mask[idx, prev_action] = 1.0

    def train(self):
        """
        train the deep q-learning network.
        """
        # start training only when there are enough experiences.
        if self._experiences_num < self._experiences_min / PARAM_K:
            return

        ixs = np.random.choice(self._experiences_max, self._batch_size, replace=True)

        fatches = [self._tf_loss, self._tf_training]

        feed = {
            self._tf_prev_states: self._experiences_prev_states[ixs],
            self._tf_prev_action_masks: self._experiences_actions_mask[ixs],
            self._tf_prev_rewards: self._experiences_rewards[ixs],
            self._tf_next_states: self._experiences_next_states[ixs]
        }

        loss, _ = self._tf_session.run(fatches, feed_dict=feed)

    def act(self, observation, reward, done):
        """
        ask the next action from the agent
        """
        self._time += 1

        if self._epsilon > self._epsilon_min:
            self._epsilon *= self._epsilon_decay_rate ** PARAM_K
            # print "Epsilon = ", self._epsilon

        if np.random.rand() > self._epsilon:
            states = np.array([observation])
            action_values = self._tf_action_value_predict.eval(feed_dict={self._tf_prev_states: states})
            action = np.argmax(action_values)
        else:
            action = np.random.choice(self._dim_action)

        if self._prev_state is not None:
            if done:
                observation = np.zeros_like(observation)
            self.create_experience(self._prev_state, self._prev_action, reward, observation)

        self._prev_state = None if done else observation
        self._prev_action = None if done else action

        self.train()

        return action

    def save(self, episode_count):
        saver = tf.train.Saver()
        saver.save(self._tf_session, "flappy_" + str(version) + "_" + str(episode_count) + ".ckpt")


def statify(observation):
    return observation.values()

def processReward(raw_reward):
    if raw_reward == 0.0:
        return 15.0
    elif raw_reward == 1.0:
        return 1000.0
    elif raw_reward == -5.0:
        return -1000.0

# Agent settings
network_layers = [40, 80]   

# Initialize simulation
game = FlappyBird()
p = PLE(game, fps=30, display_screen=True)
p.init()
print p.getActionSet()

# Create agent
agent = DeepQLearningAgent(len(game.getGameState()), p.getActionSet(), network_layers)

# Stats
batch_score = 0
episode_count = 0
output = open("out_" + str(version) + ".txt", "w")
batch_frame = 0
checkpoint = 1
OUTPUT_FREQ = 100

# Run simulation
while True:
#(Infinite episodes, till training terminated manually)
    episode_count += 1
    frame_count = 0
    reward = 0
    action = 0

    while True:
    #(Infinite frames, till birdy dies)
        done = p.game_over()

        if frame_count % PARAM_K == 0:
            observation = game.getGameState()
            action = agent.act(statify(observation), reward, done)
            reward = processReward(p.act(p.getActionSet()[action]))
            
        else:
            reward += processReward(p.act(p.getActionSet()[action]))
                    
        frame_count += 1
        # print "reward = ", reward

        if done:
            batch_score += p.score()
            batch_frame += frame_count
            observation = game.getGameState()
            action = agent.act(statify(observation), reward, done)
            p.reset_game()
            if episode_count % OUTPUT_FREQ == 0:
                avg_frame = batch_frame / OUTPUT_FREQ
                avg_score = batch_score / OUTPUT_FREQ
                batch_frame = 0.0
                batch_score = 0.0
                output.write("".join([str(x) for x in ["Episode ", episode_count, " Avg Score = ", avg_score, " Avg Frames = ", avg_frame, "\n"]]))
                print "Episode ", episode_count, " Avg Score = ", avg_score, " Avg Frames = ", avg_frame
            if episode_count == checkpoint:
                agent.save(episode_count)
                checkpoint *= 10
            break

