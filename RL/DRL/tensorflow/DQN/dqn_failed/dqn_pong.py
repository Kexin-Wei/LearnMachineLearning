#!/usr/bin/env python
# coding: utf-8

# # Atari Pong

# ## Import

# In[2]:


import tensorflow as tf
import gym

import matplotlib.pyplot as plt
import numpy as np

import os
import datetime
import shutil
import pickle

import imageio
import cv2

import argparse
import random

import pdb
from collections import deque

from dqn_model import CNN, ObWrapper, Agent
from gif_util import gif_save, png_save

class Args:
    EPOCHS   = 500
    
    GAMMA    = 0.9
    EPSILON  = 1.0
    EPSILON_DECAY  = 0.9997
    EPSILON_END    = 0.02
    
    LEARNING_RATE = 0.001
    
    BATCH_SIZE  = 64
    MEMORY_SIZE = 50000

    MODEL_UPDATE_STEP   = 200
    MEMORY_SAMPLE_START = 0.2
    
    WRAPPER_SIZE = 4

args = Args
# parser = argparse.ArgumentParser(description='Chang parameter for DQN')
# parser.add_argument('-ep', '--EPOCHS',        type=int,   default=2000,   help="change epochs")

# parser.add_argument('-g',  '--GAMMA',         type=float, default=0.9,    help="change gamma")
# parser.add_argument('-e',  '--EPSILON',       type=float, default=1.0,    help="change epsilon")
# parser.add_argument('-ed', '--EPSILON_DECAY', type=float, default=0.9997, help="change epsilon decay")

# parser.add_argument('-lr', '--LEARNING_RATE', type=float, default=0.001,   help="change learning rate")

# parser.add_argument('-b',  '--BATCH_SIZE',    type=int,   default=64,     help="change batch size in train")
# parser.add_argument('-m',  '--MEMORY_SIZE',   type=int,   default=50000,   help="change memorysize")


# parser.add_argument('-mp', '--MODEL_UPDATE_STEP',   type=int,   default=4000,  help="change model update step")
# parser.add_argument('-ms', '--MEMORY_SAMPLE_START', type=float, default=0.01,   help="change memory sample start as ratio of memory size")
# parser.add_argument('-w',  '--WRAPPER_SIZE',        type=int,   default=4,     help="change wrapper size")

# args = parser.parse_args()

EPOCHS = args.EPOCHS

GAMMA   = args.GAMMA
EPSILON = args.EPSILON
EPSILON_DECAY = args.EPSILON_DECAY
EPSILON_END   = args.EPSILON_END

LEARNING_RATE = args.LEARNING_RATE

BATCH_SIZE = args.BATCH_SIZE
MEMORY_SIZE = args.MEMORY_SIZE

MODEL_UPDATE_STEP = args.MODEL_UPDATE_STEP

MEMORY_SAMPLE_START = args.MEMORY_SAMPLE_START

WRAPPER_SIZE = args.WRAPPER_SIZE


env_name = 'Pong-v0'
env = gym.make(env_name)

N_ACT = env.action_space.n
N_OB  = env.observation_space.shape


# In[12]:


agent = Agent(N_ACT,N_OB,             \
              GAMMA = GAMMA, EPSILON = EPSILON, EPSILON_DECAY = EPSILON_DECAY,              \
              MODEL_UPDATE_STEP   = MODEL_UPDATE_STEP,                \
              MEMORY_SAMPLE_START = MEMORY_SAMPLE_START,              \
              LEARNING_RATE = LEARNING_RATE,              \
              MEMORY_SIZE   = MEMORY_SIZE,                \
              BATCH_SIZE    = BATCH_SIZE,                 \
              WRAPPER_SIZE  = WRAPPER_SIZE)


# =========== Main ============================================== 

reward_summary = {
    'max':[],
    'min':[],
    'ave':[],
    'sum':[]
}

ROOT_DIR = '../test_pong'
DIR = os.path.join(ROOT_DIR,datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
try:
    os.makedirs(DIR)
except:
    pass

DIR_PNG = os.path.join(DIR,"temp_png")

# =========================================================

state      = ObWrapper(WRAPPER_SIZE=WRAPPER_SIZE)
state_next = ObWrapper(WRAPPER_SIZE=WRAPPER_SIZE)

ob = env.reset()
state_next(ob)
while (len(state_next) is not WRAPPER_SIZE):
    state(ob)
    state_next(ob)

# =========================================================
for ep in range(EPOCHS):
    
    
    os.mkdir(DIR_PNG) 
    
    reward_list = []
    step = 0
    ob = env.reset()
    
    ob, reward, done,info = env.step(1) #fire
    
    if agent.EPSILON > EPSILON_END:
        agent.EPSILON *=agent.EPSILON_DECAY
    
    log_file = open(DIR+'/log.txt','a')

    loss =[]
    accuracy = []
    #======================================================
    while(1):
        png_save(DIR_PNG,env,step)
        
        state(ob)
        act = agent.get_action(state)
        
        ob_next, reward, done, info = env.step(act)
        
        #reward = 10 if reward else -1
        state_next(ob_next)
        
        agent.memo_append([state.packup(), act, reward, state_next.packup(), done])
        history = agent.train()
        agent.target_model_update()
        if history:
            loss.append(history['loss'][0])
            accuracy.append(history['accuracy'][0])
        
        ob = ob_next
        step += 1
        reward_list.append(reward)
        
        if done:
            out = "Epoch {}\t - average rewards {} - step {}".format(ep,sum(reward_list)/len(reward_list),step)
            log_file.write(out+"\n")
            if len(loss):
                log_file.write("\t\t - ave loss {} - ave accuracy {}\n".format(ep,sum(loss)/len(loss),sum(accuracy)/len(accuracy)))
            reward_summary['max'].append(max(reward_list))
            reward_summary['min'].append(min(reward_list))
            reward_summary['sum'].append(sum(reward_list))
            reward_summary['ave'].append(sum(reward_list)/len(reward_list))
            break

    gif_save(DIR,"temp_png",ep,sum(reward_list)/len(reward_list))

log_file.close()
#=============== Show the reward every ep
print('Show the reward every ep')
plt.figure()
plt.plot(reward_summary['max'],label='max')
plt.plot(reward_summary['min'],label='min')
plt.plot(reward_summary['ave'],label='ave')
plt.legend(loc=2)
plt.savefig(DIR + '/rewards.png')
   

print('Test the final round')
# observe the final run
DIR_FINAL = os.path.join(DIR,'final')
try:
    os.mkdir(DIR_FINAL)
except:
    pass

log_file = open(DIR+'/log.txt','a')

ob = env.reset()
reward_list = []
step = 0

state      = ObWrapper(WRAPPER_SIZE=WRAPPER_SIZE)
state_next = ObWrapper(WRAPPER_SIZE=WRAPPER_SIZE)
state_next(ob)

while (len(state_next) is not WRAPPER_SIZE):
    state(ob)
    state_next(ob)

while(1):
    png_save(DIR_FINAL,env,step)
    state(ob)
    
    act = np.argmax(agent.get_q_value(state))

    ob_next,reward,done,info = env.step(act)
    state_next(ob_next)
    
    reward_list.append(reward)
    step +=1
    ob = ob_next
    
    if done:
        out = 'Final: ave rewards - {}, step - {}\n'.format(sum(reward_list)/len(reward_list),step)
        log_file.write(out+"\n")
        print(out)
        break

log_file.close()       
gif_save(DIR,'final','final',sum(reward_list)/len(reward_list))
env.close()
print("Done")
