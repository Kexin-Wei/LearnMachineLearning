import cv2
import os
import imageio
import datetime
import random

import gym
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from collections import deque

from tensorflow import keras
from tensorflow.keras import layers

class NN:
    def __init__(self,N_ACT,N_OB,LEARNING_RATE = 0.01):
        self.learning_rate = LEARNING_RATE
        self.input_shape   = (2,N_OB[0])
        self.output_shape   = N_ACT
        
        self.model = self.create_nn()
        
    def create_nn(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape =self.input_shape),
            tf.keras.layers.Dense(32,activation = 'relu'),
            tf.keras.layers.Dense(32,activation = 'relu'),
            tf.keras.layers.Dense(self.output_shape, activation = 'linear')
        ])
        model.compile(
            loss = 'mse',
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics = ['accuracy']
        )
        return model

class Replay:
    def __init__(self, MEMORY_SIZE = 5000, BATCH_SIZE = 64):
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

class Agent(Replay,NN):
    def __init__(self, N_ACT,N_OB,                \
                    GAMMA   = 0.9,                \
                    EPSILON = 0.3,                \
                    MODEL_UPDATE_STEP   = 200,    \
                    MEMORY_SAMPLE_START = 20,     \
                    LEARNING_RATE = 0.01,         \
                    MEMORY_SIZE  = 10_000,        \
                    BATCH_SIZE   = 64):
        
        Replay.__init__(self, MEMORY_SIZE = MEMORY_SIZE, \
                        BATCH_SIZE = BATCH_SIZE)
        NN.__init__(self, N_ACT, N_OB, \
                     LEARNING_RATE = LEARNING_RATE)
        self.N_ACT = N_ACT
        self.GAMMA   = GAMMA
        self.EPSILON = EPSILON

        self.target_model = self.create_nn()
        self.target_model.set_weights(self.model.get_weights())
        
        self.MODEL_UPDATE_STEP = MODEL_UPDATE_STEP
        self.STEP = 0
        
        self.MEMORY_SAMPLE_START = MEMORY_SAMPLE_START
        
    def get_q_value(self, state):
        # mountain car state = [ob,ob_next]
        return self.model.predict(np.expand_dims(state,0))
    
    def get_action(self,state): 
        # get action with epsilon greedy
        q = self.get_q_value(state)
        if np.random.rand() < self.EPSILON:            
            return np.random.randint(self.N_ACT), np.amax(q)
        return np.argmax(q), np.amax(q)
    
    def train(self): 
        #if the momery len < its thershold
        if self.memo_len() < self.MEMORY_SAMPLE_START:
            return
        
        batch_memo = self.sample()
        
        # model for q now
        batch_state = np.array([ a_set_memo[0] for a_set_memo in batch_memo])
        batch_q     = self.model.predict(batch_state)
        
        # target_model for max q
        batch_state_next = np.array([ a_set_memo[3] for a_set_memo in batch_memo])
        batch_q_next = self.target_model.predict(batch_state_next)
        
        batch_q_new = []
        
        for index,(state, action, reward, state_next, done) in enumerate(batch_memo):
            if done:
                q_new = reward
            else:
                q_new = reward + self.GAMMA * max(batch_q_next[index])
            
            q = batch_q[index]
            q[action] = q_new
            # TODO: maybe add a q offset bound in [-1,1]
            batch_q_new.append(q)
            
        self.STEP +=1
        history = self.model.fit(batch_state,np.array(batch_q_new),batch_size = self.BATCH_SIZE, verbose = 0)
        return history.history
    
    def target_model_update(self):
        if self.STEP < self.MODEL_UPDATE_STEP:
            return
        self.STEP = 0
        self.target_model.set_weights(self.model.get_weights())


class Args:
    EPOCHS   = 300
    EPOCHS_T = 200
    #FRAME_END   = 500
    GAMMA    = 0.99

    EPSILON  = 1.0
    EPSILON_END    = 0.01

    
    LEARNING_RATE = 0.01
    
    BATCH_SIZE  = 32
    
    MEMORY_SIZE = 20000
    MEMORY_SAMPLE_START = 1000
    
    MODEL_UPDATE_STEP   = 400

args = Args
EPOCHS = args.EPOCHS
EPOCHS_T  = args.EPOCHS_T

GAMMA   = args.GAMMA
EPSILON = args.EPSILON
EPSILON_END   = args.EPSILON_END

LEARNING_RATE = args.LEARNING_RATE

BATCH_SIZE = args.BATCH_SIZE
MEMORY_SIZE = args.MEMORY_SIZE
MEMORY_SAMPLE_START = args.MEMORY_SAMPLE_START

MODEL_UPDATE_STEP = args.MODEL_UPDATE_STEP

env = gym.make('MountainCar-v0')
ob = env.reset()

N_ACT = env.action_space.n
N_OB  = env.observation_space.shape
print(env.action_space.n)
print(env.observation_space.shape)

agent = Agent(N_ACT,N_OB, \
              GAMMA = GAMMA, EPSILON = EPSILON, \
              MODEL_UPDATE_STEP   = MODEL_UPDATE_STEP, \
              MEMORY_SAMPLE_START = MEMORY_SAMPLE_START, \
              LEARNING_RATE = LEARNING_RATE, \
              MEMORY_SIZE   = MEMORY_SIZE, \
              BATCH_SIZE    = BATCH_SIZE)

ROOT_DIR = '../../test_car'
DIR = os.path.join(ROOT_DIR,datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
DIR_GIF = os.path.join(DIR,'gif')
DIR_MODEL = os.path.join(DIR,'Model')
try:
    os.makedirs(DIR)
    os.makedirs(DIR_GIF)
    os.mkdir(DIR_MODEL)
except:
    pass



log_file = open(DIR+'/log.txt','a')
log_file.write("EPOCH_END:{} \t FRAME_T:{}\n".format(EPOCHS,EPOCHS_T))
log_file.write("GAMMA:{} \t EPSILON:{} \t EPSILON_END:{} \t LEARNING_RATE:{}\n".format(GAMMA,EPSILON,EPSILON_END,LEARNING_RATE))
log_file.write("MEMORY_SIZE:{} \t MEMORY_SAMPLE_START:{}\n".format(MEMORY_SIZE,MEMORY_SAMPLE_START))
log_file.write("MODEL_UPDATE_STEP:{} \t BATCH_SIZE:{}\n".format(MODEL_UPDATE_STEP,BATCH_SIZE))
log_file.close()


reward_summary = []
history_summary = {
    'loss':[],
    'accuracy':[]
}

max_q_summary = []

epsilon_decay = (EPSILON-EPSILON_END)/EPOCHS_T

for ep in range(EPOCHS):
    loss        = []
    accuracy    = []
    reward_list = []
    max_q_list  = []
    images      = []
    

    if agent.EPSILON > EPSILON_END:
        agent.EPSILON -= epsilon_decay
    else:
        agent.EPSILON = EPSILON_END

    
    ob = env.reset()
    state = np.stack([ob,ob])

    ob_max=ob[0]

    log_file = open(DIR+'/log.txt','a')

    step = 0
    while(1):            
        if ep%50==0:
            images.append(cv2.resize(env.render(mode='rgb_array'),(300,200),cv2.INTER_AREA))
        
        act,max_q = agent.get_action(state)
        #print("act",act)
        ob_next, reward, done, info = env.step(act)
        # if done:
        #     if step<=200:
        #         reward = 40
        state_next = np.stack([ob,ob_next])
        agent.memo_append([state, act, reward, state_next, done])
        
        history = agent.train()
        agent.target_model_update()
        
        if history:
            loss.append(history['loss'][0])
            accuracy.append(history['accuracy'][0])

        ob_max = max(ob_max,ob_next[0])
        max_q_list.append(max_q)
        reward_list.append(reward)
        state = state_next
        step +=1
        if done:
            break
    
    if ep%50==0:
        agent.model.save(DIR_MODEL)
    reward_summary.append(sum(reward_list)/len(reward_list))
    max_q_summary.append(sum(max_q_list)/len(max_q_list))
    
    out = "\nEpoch {} \tsteps:{} \tepsilon: {:0.5f} \tave_rewards: {:0.2f} \tave_max_q:{:0.2f} \tmax_pos:{:0.2f}\t".format(ep,step,agent.EPSILON,reward_summary[-1],max_q_summary[-1],ob_max)
    log_file.write(out)
    print(out,end=" ")
    
    if len(loss):
        history_summary['loss'].append(sum(loss)/len(loss))
        history_summary['accuracy'].append(sum(accuracy)/len(accuracy))
        out = "ave loss: {:e} \tave accuracy: {:e}\t".format(history_summary['loss'][-1],history_summary['accuracy'][-1])
        log_file.write(out)
        print(out,end=" ")
    
    log_file.close()
    if len(images):
        imageio.mimsave(os.path.join(DIR_GIF,str(ep)+'_step_'+str(step)+'_r_'+str(reward_summary[-1])+'.gif'),images,fps=60)
    
    
    
print('Show the reward every ep')
plt.figure()
plt.plot(reward_summary)
plt.title('ave reward')
plt.savefig(DIR + '/rewards_ave.png')
plt.figure()
plt.plot(history_summary['loss'],label='loss')
plt.plot(history_summary['accuracy'],label='accuracy')
plt.legend(loc=2)
plt.savefig(DIR + '/loss_accuracy.png')
plt.figure()
plt.plot(max_q_summary)
plt.title("Ave Max Q Value")
plt.savefig(DIR + '/ave_max_q.png')

#input("Start Test Final?")
ob = env.reset()
state = np.stack([ob,ob])

images = []
reward_list =[]
step = 0
while(1):
    images.append(env.render(mode='rgb_array'))
    act = np.argmax(agent.get_q_value(state))
    
    ob_next,reward,done,info = env.step(act)
    state_next = np.stack([ob,ob_next])

    reward_list.append(reward)
    step +=1
    state = state_next
    
    if done :
        break
imageio.mimsave(os.path.join(DIR,'final_step_'+str(step)+'_r_'+str(sum(reward_list)/sum(reward_list))+'.gif'),images,fps=60)

print("Done")