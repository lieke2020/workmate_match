# -*- coding: utf-8 -*-
""" Deep WorkMATe with matchnet implementation
Created on Tue Nov 30 11:22:34 2021

Main file that holds 
1) The learning of a neural comparator in isolation 
2) The learning of the comparator within a Deep Reinforcement Learning agent
This is phase 1 in a 2-phase learning paradigm (Details in README)

For simplicity, the second learning phase (Reward-driven learning), is 
not included here. 

@author: Lieke Ceton
"""

#%% Import modules
import numpy as np

import workmate_match
import world
from matchnet import run_ncomp, plot_ncomp_eval
from m_runner import run_match, plot_eval, plot_perf_match

#%% Replication results Neural Comparator 

#This is a one-layer simplification of the original comparator module
#Below the comparator is trained for a given t_max steps,
#inputting random vectors with length N, which are the exact same in match trials
#(identity encoding, default) or are linearly transformed (linear encoding)

t_max = 10000             #number of training steps (10**7 in the paper)
t_eval = 1000             #number of steps that are evaluated (10**6 in paper)
N = 30                    #length of the input vector
encoding = 'identity'     #can be identity or linear

#Train the matchnet
matchnet, matchnet_data, eval_dt = run_ncomp(t_max, t_eval, N, encoding)

#Plot the evaluation data
plot_ncomp_eval(eval_dt)

#%% Phase 1: Motor babbling

#This section implements the above neural comparator in a Deep Reinforcement
#Learning agent and trains the match network in an embedded fashion. Each memory
#block has its own independent match network.
#The inputs are now transformed CIFAR-10 images and encoding is always identity
#Evaluation is extended to performance in time based on the prediction of 
#a logistic regression model, this gives a score for how easy match and non-match
#trials can be separated and gives the cue for the end of training.

n = 5               #Number of agents trained
n_mem = 2           #Number of memory blocks+matching networks  per agent
n_eval = 2          #Agent selected for evaluation example
stimulus = 'cifar'  #Stimulus type can be cifar or alpha

#Initialise DMS environment --> crucial for embedding the match network
dms = world.DMS(n_stim=3, n_switches=5, stim_type = stimulus)

#Initialise buffers
i_conv_buff = np.zeros(n)   #indices of convergence
perf_match_buff = []        #matching performance for blocks of trials

for i in range(n):
    #Initialise agent 
    agent = workmate_match.WorkMATe(dms, nblocks=n_mem)  
    #Train the matching network
    agent, val_buff, i_conv_buff[i], perf_match = run_match(agent, dms) 
    #Add the performance to the buffer
    perf_match_buff.append(perf_match)
    #Return the trained agent, evaluation buffer, index of convergence, performance
    
    if i == n_eval: #Save the value buffer of the selected agent
        n_eval_data = val_buff.copy() 

#EVALUATE TRAINING: Logistic regression performance in time
#i_conv is the index of convergence (performance all n > 99%)
plot_perf_match(perf_match_buff, i_conv_buff) 

#EVALUATION EXAMPLE: Plot batch of match value outcomes after training
#m indicates the selected memory block
#A cosine similarity of 1 indicates match trials
plot_eval(val_buff, m=0) 


