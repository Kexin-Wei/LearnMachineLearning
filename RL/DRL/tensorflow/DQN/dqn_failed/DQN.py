#!/usr/bin/env python
# coding: utf-8

# # Atari Breakout DQN
import pdb
import tensorflow as tf
import matplotlib.pyplot as plt
import gym
import os
import datetime
import imageio
import numpy as np
import argparse
import pickle
import shutil
from DQNAgent import DQN_Agent

#========= parameter 
parser = argparse.ArgumentParser(description='Chang parameter for DQN')
parser.add_argument('-ep', '--EPOCHS',      type=int,   default=2000,   help="change epochs")
parser.add_argument('-e',  '--EPSILON',     type=float, default=0.3,    help="change epsilon")
parser.add_argument('-ed', '--EPSILON_DC',  type=float, default=0.9997, help="change epsilon decay")
parser.add_argument('-b',  '--BATCH_SIZE',  type=int,   default=32,     help="change batch size in train")
parser.add_argument('-m',  '--MEMORY_SIZE', type=int,   default=2000,   help="change memorysize")
args = parser.parse_args()

EPOCHS = args.EPOCHS
EPSILON = args.EPSILON
EPSILON_DC = args.EPSILON_DC
BATCH_SIZE = args.BATCH_SIZE
MEMORY_SIZE = args.MEMORY_SIZE

#============== initial
env_name = 'Breakout-v0'
env = gym.make(env_name)

N_ACT = env.action_space.n
N_OB  = env.observation_space.shape

#============= make dir path for log and figure

ROOT_DIR = "../../gym_graph"
DIR = os.path.join(ROOT_DIR,datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(DIR)

log_file = open(DIR+'/log_'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),'w')

DIR_PNG = os.path.join(DIR,'temp')

#================ env run
agent = DQN_Agent(N_ACT,N_OB, MEMORY_SIZE = MEMORY_SIZE, \
                  BATCH_SIZE = BATCH_SIZE, EPSILON = EPSILON, EPSILON_DC = EPSILON_DC)

reward_summary = {
    'max':[],
    'min':[],
    'ave':[],
    'sum':[]
}


for ep in range(EPOCHS):
    
    ob = env.reset()
    
    agent.EPSILON = agent.EPSILON*agent.EPSILON_DC
    all_reward = []
    step = 0
    
    os.makedirs(DIR_PNG)
    while(1):
        # render monitoring          
        plt.imsave(os.path.join(DIR_PNG,str(step)+'.png'),env.render(mode='rgb_array'))
        
        # take action
        act = agent.take_action(ob)
        
        # env step
        ob_next, reward, done, info = env.step(act)
            # reward modified
            # reward = reward if done else -1
        reward = 200 if reward else -1
        # memorize: sars(a) : [ob, act, reward, ob_next, done]
        agent.memorize([ob, act, reward, ob_next, done])
        
        # q-value update
        if len(agent.replay_memory) > (agent.MEMORY_SIZE/10):
            agent.train()            
            if step % 50 == 0:
                #set_trace()
                agent.target_model_update()
            
        ob = ob_next
        all_reward.append(reward)
        step += 1
        
        if done:
            #set_trace()
            log_file.write("Epoch {} - average rewards {} with step {}\n".format(ep,sum(all_reward)/len(all_reward),step))
            print("Epoch {} - average rewards {} with step {}".format(ep,sum(all_reward)/len(all_reward),step))
            reward_summary['max'].append(max(all_reward))
            reward_summary['min'].append(min(all_reward))
            reward_summary['sum'].append(sum(all_reward))
            reward_summary['ave'].append(sum(all_reward)/len(all_reward))
            break
    images = []
    for f in os.listdir(DIR_PNG):
        images.append(imageio.imread(os.path.join(DIR_PNG,f)))
    imageio.mimsave(os.path.join(DIR,str(ep)+'.gif'),images)
    shutil.rmtree(DIR_PNG)
#=============== Show the reward every ep
print('Show the reward every ep')
plt.figure()
plt.plot(reward_summary['max'],label='max')
plt.plot(reward_summary['min'],label='min')
plt.plot(reward_summary['ave'],label='ave')
plt.legend(loc=2)
plt.savefig(DIR+'/rewards.png')
#================== Store the model and reward
print('Store the model and reward')
# In[ ]:
with open(DIR+'/rewards.pickle','wb') as f:
    pickle.dump(reward_summary,f)
# agent.model.save('model'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

#=================== Test the final round
print('Test the final round')
# observe the final run
ob = env.reset()
all_reward = []
step = 0
DIR_FINAL = os.path.join(DIR,'final')
os.makedirs(DIR_FINAL)
while(1):
    plt.imsave(os.path.join(DIR_FINAL,str(step)+'.png'),env.render(mode='rgb_array'))
    act = np.argmax(agent.get_q(ob))
    
    ob_next,reward,done,infor = env.step(act)
    
    all_reward.append(reward)
    step +=1
    ob = ob_next
    
    if done:
        log_file.write('Final: ave rewards - {}, step - {}\n'.format(sum(all_reward)/len(all_reward),step))
        print('Final: ave rewards - {}, step - {}'.format(sum(all_reward)/len(all_reward),step))
        break
        
images = []
for f in os.listdir(DIR_FINAL):
    images.append(imageio.imread(os.path.join(DIR_FINAL,f)))
imageio.mimsave(os.path.join(DIR,'final.gif'),images)
shutil.rmtree(DIR_FINAL)
       
log_file.close()
env.close()
print("Done")
