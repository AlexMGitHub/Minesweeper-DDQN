#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 13:49:39 2020

@author: alex

###############################################################################
# train_minesweeper.py
#
# Revision:     1.00
# Date:         6/03/2020
# Author:       Alex
#
# Purpose:      Train an agent to play Minesweeper using 
#               a Double Deep Q-Network.
#
# Notes:        A solve condition can be set to stop training once the average 
#               score exceeds the user-defined threshold.
#
##############################################################################
"""


# %% Imports
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.optimizers import Adam
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
from minesweeper import Minesweeper
from DDQN import DoubleDQNAgent


# %% Functions
def create_dqn(LR_INITIAL):
    # Create a CNN to act as a function approximator for deep Q-learning
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', input_shape = (ROWDIM, COLDIM, 9), 
                          activation = 'relu', use_bias = True, data_format='channels_last'))
    model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu', use_bias = True))
    model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu', use_bias = True))
    model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu', use_bias = True))
    model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu', use_bias = True))
    model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu', use_bias = True))
    model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu', use_bias = True))
    model.add(Conv2D(1, (1, 1), padding='same', activation = 'linear', use_bias = True))
    model.add(Flatten())
    model.compile(loss='mse', optimizer=Adam(lr=LR_INITIAL))
    return model

def create_timestamp():
    timestamp = datetime.now(tz=None)
    timestamp_str = timestamp.strftime("%d-%b-%Y(%H:%M:%S)")
    return timestamp_str

def holdout_predicted_q(holdout_states, agent):
    # Periodically average the maximum predicted Q-value on a set of holdout states
    qval_list = []
    qval_list.append(np.amax(agent.online_network.predict(holdout_states), axis=1))
    return np.mean(qval_list)

def moving_average(a, n=100) :
    ret = np.cumsum(a, dtype=np.float64)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_trial(trial):
    ave = moving_average(trial, MOVING_AVE_WINDOW)
    plt.plot(trial, label='Episode Score')
    plt.plot(np.arange(MOVING_AVE_WINDOW,len(trial)+1),ave, \
             label='{} Episode Moving Average'.format(MOVING_AVE_WINDOW))
    plt.plot(np.arange(1,len(trial)+1),SOLVE_CONDITION*np.ones(len(trial)),\
              label='{} point average goal'.format(SOLVE_CONDITION),linestyle='dashed', color='r')
    plt.ylabel('Episode Score')
    plt.xlabel('Episode Number')
    plt.title('Double DQN {} Training Progress'.format(ENV_NAME))
    plt.legend()
    plt.show()

def plot_holdout_states(holdout_states_q):
    # Plot the average Q of the holdout states over the course of training
    plt.plot(holdout_states_q, label='Holdout States Average Q')
    plt.ylabel('Average Action Value (Q)')
    plt.xlabel('Training Epochs')
    plt.title('Holdout States Average Q During Training')
    plt.show()


# %% Initialize game environment and settings
ENV_NAME = 'Minesweeper'
ROWDIM = 16 # Number of rows in the Minesweeper grid
COLDIM = 30 # Number of columns in the Minesweeper grid
MINE_COUNT = 99
env = Minesweeper(ROWDIM, COLDIM, MINE_COUNT)
env.seed(1)


# %%  Agent/Network Hyperparameters
LR_PIECEWISE = [0.004,0.001,0.001,0.0005,0.0005,0.00025] # NN learning rates to decay piecewise 
LR_DECAY_STEPS = [0,1e6,4e6,8e6,10e6,12e6] # Number of steps that define piecewise segments
GAMMA = 0.99 # Discount factor
EPSILON_INITIAL = 1 # Exploration rate
EPSILON_DECAY = .99
EPSILON_MIN = 0.0
TAU = 1 # Target network soft update, set to 1 to copy online network
# Experience replay parameters
EXPERIENCE_REPLAY_BATCH_SIZE = 1024
AGENT_MEMORY_LIMIT = EXPERIENCE_REPLAY_BATCH_SIZE*100
NUM_HOLDOUT_STATES = EXPERIENCE_REPLAY_BATCH_SIZE
# Prioritized Experience Replay (PER) parameters
PER_ALPHA = 0.6 # Exponent that determines how much prioritization is used
PER_BETA_MIN = 0.4 # Starting value of importance sampling correction
PER_BETA_MAX = 1.0 # Final value of beta after annealing
PER_BETA_ANNEAL_STEPS = 10e6 # Number of steps to anneal beta over
PER_EPSILON = 0.01 # Small positive constant to prevent zero priority

# Pass hyperparameters to DDQNAgent as dictionary
agent_kwargs = {
    'ROWDIM' : ROWDIM,
    'COLDIM' : COLDIM,
    'LR_PIECEWISE' : LR_PIECEWISE,
    'LR_DECAY_STEPS' : LR_DECAY_STEPS,
    'GAMMA' : GAMMA, 
    'EPSILON_INITIAL' : EPSILON_INITIAL, 
    'EPSILON_DECAY' : EPSILON_DECAY,
    'EPSILON_MIN' : EPSILON_MIN,
    'TAU' : TAU,
    'EXPERIENCE_REPLAY_BATCH_SIZE' : EXPERIENCE_REPLAY_BATCH_SIZE,
    'AGENT_MEMORY_LIMIT' : AGENT_MEMORY_LIMIT,
    'NUM_HOLDOUT_STATES' : NUM_HOLDOUT_STATES,
    'PER_ALPHA' : PER_ALPHA,
    'PER_BETA_MIN' : PER_BETA_MIN,
    'PER_BETA_MAX' : PER_BETA_MAX,
    'PER_BETA_ANNEAL_STEPS' : PER_BETA_ANNEAL_STEPS,
    'PER_EPSILON' : PER_EPSILON
    }

    
# %% Training parameters
trials = []
NUMBER_OF_TRIALS=1
MAX_TRAINING_EPISODES = 100000
MAX_STEPS_PER_EPISODE = ROWDIM*COLDIM-MINE_COUNT
SOLVE_CONDITION = 365 # Average score training will stop at if reached
MOVING_AVE_WINDOW = 100 # Number of episodes to average over
TRAIN_NETWORK_STEPS = EXPERIENCE_REPLAY_BATCH_SIZE/2 # Interval in steps before training neural network
MIN_MEMORY_FOR_EXPERIENCE_REPLAY = 2*EXPERIENCE_REPLAY_BATCH_SIZE
UPDATE_TARGET_STEPS = 80 * TRAIN_NETWORK_STEPS # Number of steps before updating target network
HOLDOUT_EPOCH = 200*TRAIN_NETWORK_STEPS # Number of agent steps between holdout state evaluations


# %% Training Loop
for trial_index in range(NUMBER_OF_TRIALS):
    online_network = create_dqn(LR_PIECEWISE[0])
    target_network = create_dqn(LR_PIECEWISE[0])
    # Uncomment lines below to resume training on an existing model
    # online_network = load_model('model/Minesweeper_Online_83754_episodes_03-Jun-2020(16:06:30).h5')  
    # target_network = load_model('model/Minesweeper_Target_83754_episodes_03-Jun-2020(16:06:30).h5')  
    agent = DoubleDQNAgent(online_network, target_network, **agent_kwargs)
    trial_episode_scores = []
    holdout_states_q = []
    avg_holdout_q = 0
    
    for episode_index in range(1, MAX_TRAINING_EPISODES+1):
        state = env.reset()
        for step_num in range(0, MAX_STEPS_PER_EPISODE):
            action, nn_state, _ = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done, nn_state)
            state = next_state
            agent.steps += 1
            
            if agent.steps == NUM_HOLDOUT_STATES:
                holdout_states = np.squeeze(np.array(agent.holdout_states))
                avg_holdout_q = holdout_predicted_q(holdout_states, agent)
                holdout_states_q.append(avg_holdout_q)
            if agent.memory_length >= MIN_MEMORY_FOR_EXPERIENCE_REPLAY:
                if agent.steps % TRAIN_NETWORK_STEPS == 0:
                    agent.experience_replay()
                if agent.steps % UPDATE_TARGET_STEPS == 0:
                    agent.update_target_network()
                if agent.steps % HOLDOUT_EPOCH == 0:
                    avg_holdout_q = holdout_predicted_q(holdout_states, agent)
                    holdout_states_q.append(avg_holdout_q)
                agent.update_beta() # Anneal PER Beta for IS weights
            if done:
                break
        
        episode_score = env.score
        trial_episode_scores.append(episode_score)
        if agent.memory_length >= MIN_MEMORY_FOR_EXPERIENCE_REPLAY:
            agent.update_epsilon() # Decay Epsilon-Greedy
        moving_avg = np.mean(trial_episode_scores[-MOVING_AVE_WINDOW:])
        result = 'loss' if env.explosion else 'win'
        print('T %d E %d scored %d (%s), avg %.2f, avg q %.2f, epsilon %.3f, lr %.3E' \
              % (trial_index,episode_index, episode_score, result, moving_avg,\
                 avg_holdout_q, agent.epsilon, agent.lrate))
        if len(trial_episode_scores) >= MOVING_AVE_WINDOW and moving_avg >= SOLVE_CONDITION: 
            print('Trial %d solved in %d episodes!' % (trial_index, episode_index))
            agent.save_model_to_disk(ENV_NAME, str(episode_index), create_timestamp())
            break
    
    if moving_avg < SOLVE_CONDITION:
        agent.save_model_to_disk(ENV_NAME, str(episode_index), create_timestamp())
    trials.append(np.array(trial_episode_scores))
    plot_trial(trials[trial_index])
    plot_holdout_states(holdout_states_q)