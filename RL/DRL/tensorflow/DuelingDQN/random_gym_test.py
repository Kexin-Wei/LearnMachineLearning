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

import pdb
from collections import deque

from gif_util import gif_save, png_save


class Args:
    FRAME_END = 10_000
    FRAME_T   = 10_000

    GAMMA       = 0.9
    EPSILON     = 1.0
    EPSILON_END = 0.02
    
    LEARNING_RATE = 0.0001
    
    BATCH_SIZE  = 32
    
    MEMORY_SIZE = 100_000
    MEMORY_SAMPLE_START = 10_000
    
    MODEL_UPDATE_STEP   = 1000
    
    WRAPPER_SIZE = 4

args = Args
# parser = argparse.ArgumentParser(description='Chang parameter for DQN')
# parser.add_argument('-ep', '--EPOCHS',        type=int,   default=2000,   help="change epochs")

# parser.add_argument('-g',  '--GAMMA',         type=float, default=0.9,    help="change gamma")
# parser.add_argument('-e',  '--EPSILON',       type=float, default=1.0,    help="change epsilon")
# parser.add_argument('-ft', '--FRAME_T',       type=float, default=0.9997, help="change frame threshold")

# parser.add_argument('-lr', '--LEARNING_RATE', type=float, default=0.001,   help="change learning rate")

# parser.add_argument('-b',  '--BATCH_SIZE',    type=int,   default=64,     help="change batch size in train")
# parser.add_argument('-m',  '--MEMORY_SIZE',   type=int,   default=50000,   help="change memorysize")


# parser.add_argument('-mp', '--MODEL_UPDATE_STEP',   type=int,   default=4000,  help="change model update step")
# parser.add_argument('-ms', '--MEMORY_SAMPLE_START', type=float, default=0.01,   help="change memory sample start as ratio of memory size")
# parser.add_argument('-w',  '--WRAPPER_SIZE',        type=int,   default=4,     help="change wrapper size")

# args = parser.parse_args()

FRAME_END = args.FRAME_END

GAMMA   = args.GAMMA
EPSILON = args.EPSILON
EPSILON_END   = args.EPSILON_END

LEARNING_RATE = args.LEARNING_RATE

BATCH_SIZE = args.BATCH_SIZE
MEMORY_SIZE = args.MEMORY_SIZE

MODEL_UPDATE_STEP = args.MODEL_UPDATE_STEP

MEMORY_SAMPLE_START = args.MEMORY_SAMPLE_START

WRAPPER_SIZE = args.WRAPPER_SIZE

FRAME_T  = args.FRAME_T

env_name = 'CartPole-v0'
env = gym.make(env_name)

N_ACT = env.action_space.n
N_OB  = env.observation_space.shape


# =========== Main ============================================== 

reward_summary = {
    'max':[],
    'min':[],
    'ave':[],
    'sum':[]
}

# =========================================================

ROOT_DIR = '../test_cartpole_random'
DIR = os.path.join(ROOT_DIR,datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
try:
    os.makedirs(DIR)
except:
    pass

PNG_NAME = "temp_png"
DIR_PNG = os.path.join(DIR,PNG_NAME)


log_file = open(DIR+'/log.txt','a')
log_file.write("FRAME_END:{}, GAMMA:{}, EPSILON:{}, FRAME_T:{}, EPSILON_END:{}, LEARNING_RATE:{}\n".format(FRAME_END,GAMMA,EPSILON,FRAME_T,EPSILON_END,LEARNING_RATE))
log_file.write("BATCH_SIZE:{}, MEMORY_SIZE:{}, MODEL_UPDATE_STEP:{}, MEMORY_SAMPLE_START:{}, WRAPPER_SIZE:{}\n".format(BATCH_SIZE,MEMORY_SIZE,MODEL_UPDATE_STEP,MEMORY_SAMPLE_START,WRAPPER_SIZE))
log_file.close()

frame_sum = 0
ep = 0
# =========================================================
while(frame_sum<FRAME_END):

    os.mkdir(DIR_PNG) 
    log_file = open(DIR+'/log.txt','a')

    loss        = []
    accuracy    = []
    reward_list = []
    max_q_list  = []
    frame       = 0
    act_repeat  = 0

    ob = env.reset()
    
    # Fire
    act = 1
    ob,_,_,_ = env.step(act)
    #======================================================
    while(1):
        #env.render()
        png_save(DIR_PNG,env,frame)
                              
        if act_repeat == 4:
            act = env.action_space.sample()
            act_repeat =0
        
        ob_next, reward, done, info = env.step(act)

        
        ob = ob_next
        
        frame      += 1
        act_repeat += 1 
        reward_list.append(reward)
        
        #if done or info['ale.lives'] < 5:
        if done:   
            break
    ep += 1
    reward_summary['max'].append(max(reward_list))
    reward_summary['min'].append(min(reward_list))
    reward_summary['sum'].append(sum(reward_list))
    reward_summary['ave'].append(sum(reward_list)/len(reward_list))
        
    frame_sum +=frame
    
    out = "\nEpoch {} \tsum_rewards: {} \tframe: {} \tframe_sum:{:e} \t".format(ep,reward_summary['sum'][-1],frame,frame_sum)
    log_file.write(out)
    print(out,end=" ")       
        
    log_file.close()
    if reward_summary['sum'][-1]:
        gif_save(DIR,PNG_NAME,ep,reward_summary['sum'][-1])
    else:
        shutil.rmtree(DIR_PNG)

#=============== Show the reward every ep
print('Show the reward every ep')
plt.figure()
plt.plot(reward_summary['max'],label='max')
plt.plot(reward_summary['min'],label='min')
plt.plot(reward_summary['ave'],label='ave')
plt.plot(reward_summary['sum'],label='sum')
plt.legend(loc=2)
plt.savefig(DIR + '/rewards.png')
