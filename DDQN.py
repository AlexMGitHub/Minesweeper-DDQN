#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 22:02:29 2020

@author: alex

###############################################################################
# DDQN.py
#
# Revision:     1.00
# Date:         6/03/2020
# Author:       Alex
#
# Purpose:      Create a Double Deep Q-Network agent to play Minesweeper.
#
# Notes:        The piecewise-linear decay of the learning rate can be 
#               graphically validated by instantiating an agent and calling
#               the test_lrate_decay() method.
#
##############################################################################
"""


import numpy as np
import random
from SumTree import SumTree
from keras import backend as K
from matplotlib import pyplot as plt

                  
class DoubleDQNAgent:

    def __init__(self, online_network, target_network, **kwargs):
        # Initialize passed hyperparameters
        self.online_network = online_network
        self.target_network = target_network
        self.rowdim = kwargs['ROWDIM']
        self.coldim = kwargs['COLDIM']
        self.gamma = kwargs['GAMMA']
        self.epsilon = kwargs['EPSILON_INITIAL']
        self.epsilon_decay = kwargs['EPSILON_DECAY']
        self.epsilon_min = kwargs['EPSILON_MIN']
        self.tau = kwargs['TAU']
        self.batch_size = kwargs['EXPERIENCE_REPLAY_BATCH_SIZE']                        
        self.memory_limit = kwargs['AGENT_MEMORY_LIMIT']
        self.num_holdout_states = kwargs['NUM_HOLDOUT_STATES']
        self.per_alpha = kwargs['PER_ALPHA']
        self.per_beta_min = kwargs['PER_BETA_MIN']
        self.per_beta_max = kwargs['PER_BETA_MAX']
        self.per_beta_anneal_steps = kwargs['PER_BETA_ANNEAL_STEPS']
        self.per_epsilon = kwargs['PER_EPSILON']
        self.lr_piecewise = kwargs['LR_PIECEWISE']
        self.lr_decay_steps = kwargs['LR_DECAY_STEPS']
        # Initialize agent parameters
        self.steps = 0
        self.holdout_states = []
        # Piecewise-linear learning rate decay parameters
        self.lrate = self.lr_piecewise[0]
        self.lrate_decay = []
        for idx in range(0,len(self.lr_piecewise)-1):
            self.lrate_decay.append((self.lr_piecewise[idx] - self.lr_piecewise[idx+1]) \
                                    / (self.lr_decay_steps[idx+1]-self.lr_decay_steps[idx]))
        # Prioritized Experience Replay (PER) parameters
        self.beta_anneal = (self.per_beta_max - self.per_beta_min) / self.per_beta_anneal_steps
        self.per_beta = self.per_beta_min
        self.sumtree = SumTree(self.memory_limit)
        self.memory_length = 0
        
    
    def act(self, state):
        flattened_state = state.flatten()
        nn_state = self.reshape_state_for_net(state)
        # Epsilon-Greedy behavior policy
        if self.epsilon > np.random.rand():
            # Explore, but only choose hidden tiles (#9)
            valid_actions = np.where(flattened_state == 9)[0]
            return np.random.choice(valid_actions), nn_state, valid_actions
        else:
            # Exploit, but only choose hidden tiles (#9)
            valid_actions = [0 if x == 9 else 1 for x in flattened_state]
            # Predict Q-values of actions using re-shaped state
            q_values = self.online_network.predict(nn_state)
            # Use valid_actions as a mask to only allow selection of hidden tiles
            valid_qvalues = np.ma.masked_array(q_values, valid_actions)
            return np.argmax(valid_qvalues), nn_state, np.squeeze(valid_qvalues)


    def experience_replay(self):
        # The online network will SELECT the action
        select_network = self.online_network
        # The target network will EVALUATE the action's Q-value
        eval_network = self.target_network
        
        minibatch, tree_indices, weights = self._per_sample()
        minibatch_new_q_values = []

        for experience, tree_idx in zip(minibatch, tree_indices):
            state, action, reward, next_state, done, nn_state, nn_next_state = experience
            experience_new_q_values = select_network.predict(nn_state)[0]
            if done:
                q_update = reward
            else:
                valid_actions = [0 if x == 9 else 1 for x in next_state.flatten()]
                # Using the select network to SELECT action
                predicted_qvalues = select_network.predict(nn_next_state)[0]
                select_net_selected_action = np.argmax(np.ma.masked_array(predicted_qvalues, valid_actions))
                # Using the eval network to EVALUATE action
                eval_net_evaluated_q_value = eval_network.predict(nn_next_state)[0][select_net_selected_action]
                q_update = reward + self.gamma * eval_net_evaluated_q_value
            # Update sum tree with new priorities of sampled experiences 
            td_error = experience_new_q_values[action] - q_update
            td_error = np.clip(td_error, -1, 1) # Clip for stability
            priority = (np.abs(td_error) + self.per_epsilon)  ** self.per_alpha
            self.sumtree.update(tree_idx, priority)
            experience_new_q_values[action] = q_update
            minibatch_new_q_values.append(experience_new_q_values)
        minibatch_states = np.squeeze(np.array([e[5] for e in minibatch]))
        minibatch_new_q_values = np.array(minibatch_new_q_values, dtype=np.float64)
        # Apply importance sampling weights during model training
        select_network.train_on_batch(minibatch_states, minibatch_new_q_values, sample_weight=weights)
        # Decay learning rate after training
        K.set_value(select_network.optimizer.learning_rate, self.lrate_decay_callback())


    def _per_sample(self):
        # Implement proportional prioritization according to Appendix B.2.1 
        # of DeepMind's paper "Prioritized Experience Replay"
        minibatch = []
        tree_indices = []
        priorities = []
        weights = []

        # Proportionally sample agent's memory        
        samples_per_segment = self.sumtree.total() / self.batch_size
        for segment in range(0,self.batch_size):
            seg_start = samples_per_segment * segment
            seg_end = samples_per_segment * (segment + 1)
            sample = random.uniform(seg_start, seg_end)
            (tree_index, priority, experience) = self.sumtree.get(sample)
            tree_indices.append(tree_index)
            priorities.append(priority)
            minibatch.append(experience)
        
        # Calculate and scale weights for importance sampling
        min_probability = np.min(priorities) / self.sumtree.total()
        max_weight = (min_probability * self.memory_length) ** (-self.per_beta)
        for priority in priorities:
            probability = priority / self.sumtree.total()
            weight = (probability * self.memory_length) ** (-self.per_beta)
            weights.append(weight / max_weight)
            
        return minibatch, tree_indices, np.array(weights)
    

    def remember(self, state, action, reward, next_state, done, nn_state):
        # Memory includes the one-hot encoded versions of the state and next
        # state to eliminate redundant computation
        nn_next_state = self.reshape_state_for_net(next_state)
        priority = 1 # Max priority with TD error clipping
        experience = (state, action, reward, next_state, done, nn_state, nn_next_state)
        self.sumtree.add(priority, experience)
        if self.memory_length < self.memory_limit: self.memory_length += 1
        # Make copies of the initial states as a holdout set to monitor convergence
        if len(self.holdout_states) < self.num_holdout_states:
            self.holdout_states.append(nn_state)

    
    def reshape_state_for_net(self, state):
        # Reshape state into one-hot encoded array of shape: 
        # (batch_size, row_dim, col_dim, channels)
        # There are 9 channels for tile values 0-8
        # Hidden tiles are implicit as zeros in all channels

        # Making prediction on ROWDIM by COLDIM input
        batch_size = 1
        nn_input = np.zeros((batch_size,self.rowdim, self.coldim, 9))
        # Perform one-hot encoding on Minesweeper grid
        for tile_num in range(0,9):
            idx1, idx2 = np.where(state == tile_num)
            nn_input[0, idx1, idx2, tile_num] = 1
        
        return nn_input
    
    
    def save_model_to_disk(self, env, numeps, timestamp):
        self.online_network.save('model/' + env + '_Online_' + numeps + 
                                 '_episodes_' + timestamp + '.h5')
        self.target_network.save('model/' + env + '_Target_' + numeps + 
                                 '_episodes_' + timestamp + '.h5')
        print("Saved models to disk")


    def lrate_decay_callback(self):
        # Decays NN learning rate in a piecewise-linear fashion during training
        lr_ds = self.lr_decay_steps
        # Create list of conditionals for piecewise function        
        cond_list = []
        for idx in range(0, len(lr_ds)-1):
            cond_list.append(self.steps >= lr_ds[idx] and self.steps < lr_ds[idx+1])
        # Create list of functions to evaluate for each segment    
        func_list = [lambda x=self.steps, lr=a, step_offset=b, decay=c: \
                  lr - (x-step_offset) * decay \
                      for a, b, c in zip(self.lr_piecewise[0:-1], lr_ds[0:-1], self.lrate_decay)]
        func_list.append(self.lr_piecewise[-1]) # Default value if all conditions False
        
        self.lrate = float(np.piecewise(float(self.steps), cond_list, func_list))
        return self.lrate


    def update_beta(self):
        # Importance sampling exponent beta increases linearly during training
        self.per_beta = min(self.per_beta + self.beta_anneal, self.per_beta_max)
        
        
    def update_epsilon(self):
        # Random exploration rate decreases after every episode
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


    def update_target_network(self):
        # If TAU = 1 then this function simply copies the weights from the 
        # online network to the target network.  For TAU < 1 the target 
        # network's weights gradually approach the online network's weights
        online_network_weights = self.online_network.get_weights()
        target_network_weights = self.target_network.get_weights()
        layer_idx = 0
        for online_weight, target_weight in zip(online_network_weights,target_network_weights):
            updated_weight = target_weight * (1-self.tau) + online_weight * self.tau
            target_network_weights[layer_idx] = updated_weight
            layer_idx += 1
        self.target_network.set_weights(target_network_weights)
        
    
    def test_lrate_decay(self):
        # Test function to validate that the piecewise-linear decay function
        # matches the user's expectations.  Calls plot_lrate_decay() to 
        # generate 3 plots of the piecewise-linear decay function
        current_step = self.steps # Keep track of agent's current step count
        current_lrate = self.lrate # Keep track of agent's current learn rate
        lr_ds = self.lr_decay_steps
        numpts = [2, 11, 11] # Number of points per segment
        for plot_type in range(0,3):
            step_list = []
            for idx in range(0,len(lr_ds)-1):
                if plot_type == 2:
                    step_list.append(np.linspace(lr_ds[idx], lr_ds[idx+1], num=numpts[plot_type]))
                    if idx == len(lr_ds)-2:
                        step_list.append(np.linspace(lr_ds[-1], lr_ds[-1]*1.5, num=numpts[plot_type]))
                else:
                    step_list.extend(np.linspace(lr_ds[idx], lr_ds[idx+1], num=numpts[plot_type]))
                    if idx == len(lr_ds)-2:
                        step_list.extend(np.linspace(lr_ds[-1], lr_ds[-1]*1.5, num=numpts[plot_type]))
            lrate = []
            if plot_type == 2:
                for sub_list in step_list:
                    temp_list = []
                    for step in sub_list:
                       self.steps = step 
                       temp_list.append(self.lrate_decay_callback()) 
                    lrate.append(temp_list)
            else:
                for step in step_list: # Sweep through steps
                    self.steps = step 
                    lrate.append(self.lrate_decay_callback())
            # Generate plots
            self.plot_lrate_decay(plot_type, step_list, lrate)
            self.steps = current_step # Revert back to initial step count
            self.lrate = current_lrate # Revert back to initial learn rate
        return step_list, lrate
 
    
    def plot_lrate_decay(self, plot_type, step_list, lrate):
        # Plot 1: The entire pw-linear function on a semilogy scale with labels
        # Plot 2: The entire piecewise-linear decay function on a linear scale
        # Plot 3: The pw-linear segments are broken up into separate subplots
        lr_ds = self.lr_decay_steps
        if plot_type == 0: # Labeled segments on semilogy scale
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.semilogy(step_list, lrate, label='Learning Rate')
            for x,y in zip(lr_ds, self.lr_piecewise):
                  ax.annotate('({:.2E}, {:.2E})'.format(x,y), xy=(x,y), textcoords='data')
        elif plot_type == 1: # Unlabeled segments on linear scale
            fig = plt.figure()
            plt.plot(step_list, lrate, label='Learning Rate')
        elif plot_type == 2: # Labeled linear segments in separate subplots
            if len(step_list) % 2 != 0: step_list = step_list[0:-1]
            cols = 2
            rows = len(step_list) // cols
            fig, axs = plt.subplots(nrows=rows, ncols=cols)
            idx = 0
            for row in axs:
                for col in row:
                    col.plot(step_list[idx], lrate[idx])
                    idx += 1
        if plot_type == 2: # Title and ticks for sub-plots
            fig.suptitle('Learning Rate Piecewise-Linear Segments')
            for idx, ax in enumerate(axs.flat):
                try:
                    ax.set_xticks([lr_ds[idx], lr_ds[idx+1]])
                    ax.set_yticks([self.lr_piecewise[idx], self.lr_piecewise[idx+1]])
                except:
                    ax.set_xticks([lr_ds[idx], lr_ds[idx]*1.5])
                    ax.set_yticks([self.lr_piecewise[idx]*1.5, self.lr_piecewise[idx], self.lr_piecewise[idx]/2])
            fig.tight_layout(pad=3.0)
            plt.show()
        else: # Plot labels for linear and semilogy plots            
            plt.ylabel('Learning Rate')
            plt.xlabel('Steps')
            plt.title('Piecewise-Linear Learning Rate Decay Function')
            plt.show()