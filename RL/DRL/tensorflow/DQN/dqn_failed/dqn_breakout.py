#!/usr/bin/env python
# coding: utf-8

# # Atari Breakout

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

from model import CNN, ObWrapper, Agent
from gif_util import gif_save, png_save


class Args:
    EPOCHS   = 500
    
    GAMMA    = 0.99
    EPSILON  = 1.0
    EPSILON_DECAY  = 0.99
    EPSILON_END    = 0.1
    
    LEARNING_RATE = 0.00025
    
    BATCH_SIZE  = 32
    MEMORY_SIZE = 1000000

    MODEL_UPDATE_STEP   = 1000
    MEMORY_SAMPLE_START = 40
    
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


env_name = 'Breakout-v0'
env = gym.make(env_name)

N_ACT = env.action_space.n
N_OB  = env.observation_space.shape

agent = Agent(N_ACT,N_OB, \
              GAMMA = GAMMA, EPSILON = EPSILON, EPSILON_DECAY = EPSILON_DECAY, \
              MODEL_UPDATE_STEP   = MODEL_UPDATE_STEP, \
              MEMORY_SAMPLE_START = MEMORY_SAMPLE_START, \
              LEARNING_RATE = LEARNING_RATE, \
              MEMORY_SIZE   = MEMORY_SIZE, \
              BATCH_SIZE    = BATCH_SIZE, \
              WRAPPER_SIZE  = WRAPPER_SIZE)


# =========== Main ============================================== 

reward_summary = {
    'max':[],
    'min':[],
    'ave':[],
    'sum':[]
}
history_summary = {
    'loss':[],
    'accuracy':[]
}
max_q_summary = []

ROOT_DIR = '../test_break'
DIR = os.path.join(ROOT_DIR,datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
try:
    os.makedirs(DIR)
except:
    pass

DIR_PNG = os.path.join(DIR,"temp_png")


log_file = open(DIR+'/log.txt','a')
log_file.write("EPOCHS:{}, GAMMA:{}, EPSILON:{}, EPSILON_DECAY:{}, EPSILON_END:{}, LEARNING_RATE:{}\n".format(EPOCHS,GAMMA,EPSILON,EPSILON_DECAY,EPSILON_END,LEARNING_RATE))
log_file.write("BATCH_SIZE:{}, MEMORY_SIZE:{}, MODEL_UPDATE_STEP:{}, MEMORY_SAMPLE_START:{}, WRAPPER_SIZE:{}\n".format(BATCH_SIZE,MEMORY_SIZE,MODEL_UPDATE_STEP,MEMORY_SAMPLE_START,WRAPPER_SIZE))
log_file.close()

# =========================================================
for ep in range(EPOCHS):
    
    
    os.mkdir(DIR_PNG) 
    
    reward_list = []
    step = 0
    ob = env.reset()
    
    # feed  4 in obwrapper
    state      = ObWrapper(WRAPPER_SIZE=WRAPPER_SIZE)
    state_next = ObWrapper(WRAPPER_SIZE=WRAPPER_SIZE)
    state_next(ob)
    while (len(state_next) is not WRAPPER_SIZE):
        state(ob)
        state_next(ob)

    ob, reward, done,info = env.step(1) #fire
    
    if agent.EPSILON > EPSILON_END:
        agent.EPSILON *=agent.EPSILON_DECAY
    else:
        agent.EPSILON = EPSILON_END
    
    log_file = open(DIR+'/log.txt','a')

    loss =[]
    accuracy = []
    max_q_list = []
    #======================================================
    while(1):
        #env.render()
        png_save(DIR_PNG,env,step)
        
        state(ob)

        act, max_q = agent.get_action(state)
        
        ob_next, reward, done, info = env.step(act)
        #reward = 1000 if reward else -1
        state_next(ob_next)
        
        agent.memo_append([state.packup(), act, reward, state_next.packup(), done])
        
        history = agent.train()
        
        if history:
            loss.append(history['loss'][0])
            accuracy.append(history['accuracy'][0])

        agent.target_model_update()
        
        ob = ob_next
        step += 1
        
        reward_list.append(reward)
        max_q_list.append(max_q)
  
        if done or info['ale.lives'] < 5:
            reward_summary['max'].append(max(reward_list))
            reward_summary['min'].append(min(reward_list))
            reward_summary['sum'].append(sum(reward_list))
            reward_summary['ave'].append(sum(reward_list)/len(reward_list))
            
            max_q_summary.append(np.max(max_q_list))
            
            out = "\nEpoch {} \tepsilon: {} \tsum_rewards: {} \tstep: {} \tmax_q:{} \t".format(ep,agent.EPSILON,reward_summary['sum'][-1],step,max_q_summary[-1])
            print(out,end=" ")
            log_file.write(out)
            if len(loss):
                history_summary['loss'].append(sum(loss)/len(loss))
                history_summary['accuracy'].append(sum(accuracy)/len(accuracy))
                
                out = "ave loss: {}  \tave accuracy: {}".format(history_summary['loss'][-1],history_summary['accuracy'][-1])
                log_file.write(out)
                print(out,end=" ")
            break
    
    del state
    del state_next
    log_file.close()
    gif_save(DIR,"temp_png",ep,sum(reward_list)/len(reward_list))

#=============== Show the reward every ep
print('Show the reward every ep')
plt.figure()
plt.plot(reward_summary['max'],label='max')
plt.plot(reward_summary['min'],label='min')
plt.plot(reward_summary['ave'],label='ave')
plt.plot(reward_summary['sum'],label='sum')
plt.legend(loc=2)
plt.savefig(DIR + '/rewards.png')
plt.figure()
plt.plot(history_summary['loss'],label='loss')
plt.plot(history_summary['accuracy'],label='accuracy')
plt.legend(loc=2)
plt.savefig(DIR + '/loss_accuracy.png')
plt.figure()
plt.plot(max_q_summary,label='max_q')
plt.savefig(DIR + '/max_q.png')
   

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

ob, reward, done,info = env.step(1) #fire

while(1):
    #env.render()
    
    #pdb.set_trace()
    png_save(DIR_FINAL,env,step)
    state(ob)
    
    act = np.argmax(agent.get_q_value(state))

    ob_next,reward,done,info = env.step(act)
    state_next(ob_next)
    
    reward_list.append(reward)
    step +=1
    ob = ob_next
    
    if done or info['ale.lives'] < 5:
        out = 'Final:\tsum rewards: {}\tstep: {}'.format(sum(reward_list),step)
        log_file.write("\n"+out)
        print(out)
        break

log_file.close()       
gif_save(DIR,'final','final',sum(reward_list))
env.close()
print("Done")
