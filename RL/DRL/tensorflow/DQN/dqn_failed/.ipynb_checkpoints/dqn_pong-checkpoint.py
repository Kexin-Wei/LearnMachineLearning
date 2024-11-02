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
from collections import deque
from IPython.core.debugger import set_trace
import pdb


parser = argparse.ArgumentParser(description='Chang parameter for DQN')
parser.add_argument('-ep', '--EPOCHS',        type=int,   default=2000,   help="change epochs")

parser.add_argument('-g',  '--GAMMA',         type=float, default=0.9,    help="change gamma")
parser.add_argument('-e',  '--EPSILON',       type=float, default=1.0,    help="change epsilon")
parser.add_argument('-ed', '--EPSILON_DECAY', type=float, default=0.9997, help="change epsilon decay")

parser.add_argument('-lr', '--LEARNING_RATE', type=float, default=0.001,   help="change learning rate")

parser.add_argument('-b',  '--BATCH_SIZE',    type=int,   default=64,     help="change batch size in train")
parser.add_argument('-m',  '--MEMORY_SIZE',   type=int,   default=50000,   help="change memorysize")


parser.add_argument('-mp', '--MODEL_UPDATE_STEP',   type=int,   default=4000,  help="change model update step")
parser.add_argument('-ms', '--MEMORY_SAMPLE_START', type=float, default=0.01,   help="change memory sample start as ratio of memory size")
parser.add_argument('-w',  '--WRAPPER_SIZE',        type=int,   default=4,     help="change wrapper size")

args = parser.parse_args()

# ## Class
# - ObWrapper as env support
#   - ob->state
#   - function:
#     - wrapper_append(ob)
#     - wrapper_len()
#     - wrapper_packup()
# - Replay
#   - sample memory for train
#   - function:
#     - mome_append(a_set_memory)
#     - sample()
# - CNN
#   - create a cnn refer to dqn paper
# - Agent 
#   - Replay
#   - CNN

# ### ObWrapper Class

# In[3]:


# make sure import deque from collections 
# and opencv(cv2)
# and numpy as np
class ObWrapper:
    def __init__(self, WRAPPER_SIZE = 4 ):
        self.WRAPPER_SIZE = WRAPPER_SIZE
        self.s = deque([],maxlen = WRAPPER_SIZE) #wrapper how many frame together
        
    def __call__(self,ob):
        self.s.append(cv2.cvtColor(ob,cv2.COLOR_BGR2GRAY)[::2,::2][17:97,:]/255.0)

    def __len__(self):
        return len(self.s)
    
    def packup(self):
        if len(self.s) < self.WRAPPER_SIZE:
            return print("Wrapper too small, unpackable")
        a = np.array([self.s[i] for i in range(self.WRAPPER_SIZE)])
        b = np.transpose(a,(1,2,0))  # or b = np.einsum('ijk->jki',a)
        return b.reshape(-1,80,80,self.WRAPPER_SIZE)


# ### Replay Class

# In[4]:


# make sure import random
# and deque from collections
class Replay:
    def __init__(self, MEMORY_SIZE = 50_000, BATCH_SIZE = 64):
        self.BATCH_SIZE = BATCH_SIZE
        self.MEMORY_SIZE = MEMORY_SIZE
        self.memory = deque([],maxlen = MEMORY_SIZE)
        
    def memo_append(self, a_set_memory):
        # a_set_memory = sars(a) : [ob, (act), reward, ob_next, done]
        self.memory.append(a_set_memory)

    def memo_len(self):
        return len(self.memory)
        
    def sample(self):
        return random.sample(self.memory,self.BATCH_SIZE)


# ### CNN Class

# In[5]:


# import tensorflow as tf
# import numpy as np
class CNN:
    def __init__(self,N_ACT, N_OB,                 \
                 WRAPPER_SIZE = 4,                 \
                 LEARNING_RATE = 0.01):
        # N_OB: frame for weight, height, color_channel
        # -> backup to n frame : weight, height, WRAPPER_SIZE
        
        self.INPUT_SIZE  = self.input_size([80,80], WRAPPER_SIZE)
        self.OUTPUT_SIZE = N_ACT
        self.LEARNING_RATE = LEARNING_RATE
        self.model = self.create_cnn()
        
    def input_size(self,N_OB, WRAPPER_SIZE):
        return (N_OB[0],N_OB[1],WRAPPER_SIZE)
    
    def create_cnn(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16,8,strides = 4,input_shape =self.INPUT_SIZE,\
                                   activation = 'relu'),
            tf.keras.layers.Conv2D(32,4,strides = 2, activation = 'relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256,               activation = 'relu'),
            tf.keras.layers.Dense(self.OUTPUT_SIZE, activation = 'linear'),
        ])
        model.compile(
            loss = 'huber_loss',
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE),
            metrics   = ['accuracy']
        )
        return model


# ### Agent

# In[6]:


class Agent(Replay,CNN):
    def __init__(self, N_ACT,N_OB,                 \
                 GAMMA = 0.9, EPSILON = 0.3, EPSILON_DECAY = 0.997,                 \
                 MODEL_UPDATE_STEP = 2000, MEMORY_SAMPLE_START = 0.2,                 \
                 LEARNING_RATE = 0.01,                 \
                 MEMORY_SIZE = 50_000,BATCH_SIZE = 64,                 \
                 WRAPPER_SIZE = 4):
        Replay.__init__(self, MEMORY_SIZE = MEMORY_SIZE, \
                        BATCH_SIZE = BATCH_SIZE)
        
        CNN.__init__(self, N_ACT, N_OB, \
                     WRAPPER_SIZE = WRAPPER_SIZE, \
                     LEARNING_RATE = LEARNING_RATE)

        self.N_ACT   = N_ACT
        self.GAMMA   = GAMMA
        self.EPSILON = EPSILON
        self.EPSILON_DECAY = EPSILON_DECAY

        self.target_model = self.create_cnn()
        self.target_model.set_weights(self.model.get_weights())
        self.MODEL_UPDATE_STEP = MODEL_UPDATE_STEP
        self.STEP = 0
        
        self.MEMORY_SAMPLE_START = MEMORY_SAMPLE_START
        
    
    def get_q_value(self, state):
        # state is obwrapper.packup
        
        q = self.model.predict(state.packup())
        return q
    
    
    def get_action(self,state): # get action with epsilon greedy
        if np.random.rand() < self.EPSILON:
            return np.random.randint(self.N_ACT)
        return np.argmax(self.get_q_value(state))
    
    
    def train(self): 
        #if the momery len > 0.2 memory size
        if self.memo_len() < self.MEMORY_SIZE * self.MEMORY_SAMPLE_START:
            return
        batch_memo = self.sample()
        
        # model for q now
        batch_state = np.array([ a_set_memo[0][0,:,:,:] for a_set_memo in batch_memo])
        batch_q     = self.model.predict(batch_state)
        
        # target_model for max q
        batch_state_next = np.array([ a_set_memo[3][0,:,:,:] for a_set_memo in batch_memo])
        batch_q_next = self.target_model.predict(batch_state_next)
        
        batch_q_new = []
        for index,(state, action, reward, state_next, done) in enumerate(batch_memo):
            if done:
                q_new = reward
            else:
                q_new = reward + self.GAMMA * max(batch_q_next[index])
            
            q = batch_q[index]
            q[action] = q_new
            batch_q_new.append(q)
            
        self.STEP +=1
        history = self.model.fit(batch_state,np.array(batch_q_new),batch_size = self.BATCH_SIZE, verbose = 0)
        return history.history
        
    def target_model_update(self):
        if self.STEP < self.MODEL_UPDATE_STEP:
            return
        self.STEP = 0
        self.target_model.set_weights(self.model.get_weights())


# ## GIF handel

# In[7]:


def gif_save(DIR_PNG,ep,ave_reward):
    # make gif
    images = []
    for f in os.listdir(DIR_PNG):
        images.append(imageio.imread(os.path.join(DIR_PNG,f)))
    imageio.mimsave(os.path.join(DIR,str(ep)+'_r_'+str(ave_reward)+'.gif'),images)
    shutil.rmtree(DIR_PNG)


# In[8]:


def png_save(DIR_PNG,env,step):
    plt.imsave(os.path.join(DIR_PNG,str(step)+'.png'),env.render(mode='rgb_array'))


# ## Train The Model        

# In[9]:
# class Args:
#     EPOCHS   = 2
    
#     GAMMA    = 0.9
#     EPSILON  = 0.3
#     EPSILON_DECAY  = 0.9997
    
#     LEARNING_RATE = 0.01
    
#     BATCH_SIZE  = 64
#     MEMORY_SIZE = 5_00

#     MODEL_UPDATE_STEP   = 2000
#     MEMORY_SAMPLE_START = 0.2
    
#     WRAPPER_SIZE = 4


# # In[10]:


# args = Args
EPOCHS = args.EPOCHS

GAMMA   = args.GAMMA
EPSILON = args.EPSILON
EPSILON_DECAY = args.EPSILON_DECAY

LEARNING_RATE = args.LEARNING_RATE

BATCH_SIZE = args.BATCH_SIZE
MEMORY_SIZE = args.MEMORY_SIZE

MODEL_UPDATE_STEP = args.MODEL_UPDATE_STEP

MEMORY_SAMPLE_START = args.MEMORY_SAMPLE_START

WRAPPER_SIZE = args.WRAPPER_SIZE


# In[11]:


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


# ### Main

# In[13]:


reward_summary = {
    'max':[],
    'min':[],
    'ave':[],
    'sum':[]
}

ROOT_DIR = 'test_DQN'
DIR = os.path.join(ROOT_DIR,datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
try:
    os.makedirs(DIR)
except:
    pass



state      = ObWrapper(WRAPPER_SIZE=WRAPPER_SIZE)
state_next = ObWrapper(WRAPPER_SIZE=WRAPPER_SIZE)

ob = env.reset()
state_next(ob)
while (len(state_next) is not WRAPPER_SIZE):
    state(ob)
    state_next(ob)

DIR_PNG = os.path.join(DIR,"temp_png")


# In[ ]:


# =========================================================
for ep in range(EPOCHS):
    
    
    os.mkdir(DIR_PNG) 
    
    reward_list = []
    step = 0
    ob = env.reset()
    
    ob, reward, done,info = env.step(1) #fire
    
    agent.EPSILON *=agent.EPSILON_DECAY
    
    log_file = open(DIR+'/log.txt','a')
    loss_file = open(DIR+'/loss.txt','a')
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
            out = "Epoch {} - average rewards {} - step {}".format(ep,sum(reward_list)/len(reward_list),step)
            log_file.write(out+"\n")
            loss_file.write("Epoch {} - ave loss {} - ave accuracy {}\n".format(ep,sum(loss)/len(loss),sum(accuracy)/len(accuracy)))
            reward_summary['max'].append(max(reward_list))
            reward_summary['min'].append(min(reward_list))
            reward_summary['sum'].append(sum(reward_list))
            reward_summary['ave'].append(sum(reward_list)/len(reward_list))
            break

    gif_save(DIR_PNG,ep,sum(reward_list)/len(reward_list))


# In[ ]:


log_file.close()
loss_file.close()
#=============== Show the reward every ep
print('Show the reward every ep')
plt.figure()
plt.plot(reward_summary['max'],label='max')
plt.plot(reward_summary['min'],label='min')
plt.plot(reward_summary['ave'],label='ave')
plt.legend(loc=2)
plt.savefig(DIR + '/rewards.png')


# reward_summary = {[max],[min],[ave]}
# 
# for ep:
# 
#     ob = env.reset()
#     
#     reward = []
#     save the gif
#     while(done):
#         env.reder() 
#             - save png
#         act = get_action(ob)
#             
#         state<-ob
#         
#         ob_next, reward, done ,info = env.step(act)
#             reward = big if reward else -1
#             
#         state_next<-ob_next
#         
#         #=========
#         if state.len>3:
#             a-set-memory = [state,packup,(act),reward, state_next,done]
#             memo_append(a-set-memory)
#             train()
#             target_update()
#         #=========
#         
#         ob = ob_next
#         
#         reward_list.append(reward)
#         
#         if done:
#             print(step,reward)
#             log.write()
#             break
#             
#     reward_summary[max][min][ave][sum]<-reward_list
#     build_gif(ep)
#     

# ### Final Run for Test

# In[ ]:


# print(state.packup().shape,state.s[0].shape[0],state.s[0].shape[0],WRAPPER_SIZE)


# In[1]:


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
        log_file.writelines(out)
        print(out)
        break
        
gif_save(DIR_FINAL,'final',sum(reward_list)/len(reward_list))
       
log_file.close()
env.close()
print("Done")
