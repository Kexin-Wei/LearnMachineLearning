# %%
import os
import imageio
import datetime
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from collections import deque
from dm_wrapper import make_env

# %%
class Args:
    FRAME_END   = 1_000_000
    FRAME_T = 50_000
    
    ALPHA = 0.5
    BETA  = 0.5
    BASE  = 0.1
    
    GAMMA = 0.99

    EPSILON      = 1.0
    EPSILON_END  = 0.1
    
    
    LEARNING_RATE = 0.00025
    
    BATCH_SIZE  = 32
    
    MEMORY_SIZE = 100_000
    MEMORY_SAMPLE_START = 1_000
    
    MODEL_TRAIN_STEP    = 4
    MODEL_UPDATE_STEP   = 10000/MODEL_TRAIN_STEP
    
    # FRAME_END   = 500
    # FRAME_T = 100
    # MEMORY_SAMPLE_START = 100
    # MODEL_UPDATE_STEP   = 4

args = Args

# %%
class DDQN:
    # Reference:
    # https://github.com/EvolvedSquid/tutorials/blob/master/dqn/train_dqn.ipynb
    # TODO: modify for importance sampling
    def __init__(self, N_ACT,N_OB,LEARNING_RATE = 0.01):
        self.INPUT_SIZE = N_OB
        self.N_ACT      = N_ACT

        self.LEARNING_RATE = LEARNING_RATE
        self.model = self.create_dqn()

    def create_dqn(self):
        input_layer = tf.keras.layers.Input(shape=self.INPUT_SIZE)
        cnn1 = tf.keras.layers.Conv2D(32,8,strides=4,activation='relu')(input_layer)
        cnn2 = tf.keras.layers.Conv2D(64,4,strides=2,activation='relu')(cnn1)
        cnn3 = tf.keras.layers.Conv2D(64,3,strides=1,activation='relu')(cnn2)
        flatten = tf.keras.layers.Flatten()(cnn3)
        value = tf.keras.layers.Dense(1)(flatten)
        advantage = tf.keras.layers.Dense(self.N_ACT)(flatten)

        reduce_mean = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x,axis=1,keepdims=True)) # custom a type of layer
        q_output = tf.keras.layers.Add()([
            value,tf.keras.layers.Subtract()([
                advantage,reduce_mean(advantage)
                ])
            ])
        model = tf.keras.Model(input_layer,q_output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE, clipnorm = 1.0),
            loss = tf.keras.losses.Huber(),
            metrics = ['accuracy']
        )
        return model
    
# %%
class PReplay:
    # priorited replay buffer
    def __init__(self, MEMORY_SIZE = 5000, \
                       ALPHA = 0.5, \
                       BETA = 0.5,  \
                       BASE = 0.1,  \
                       BATCH_SIZE = 64):
        
        self.BATCH_SIZE  = BATCH_SIZE
        self.MEMORY_SIZE = MEMORY_SIZE
        self.ALPHA = ALPHA
        self.BETA  = BETA
        self.BASE  = BASE
        
        self.state_memo      = deque([],maxlen = MEMORY_SIZE)
        self.state_next_memo = deque([],maxlen = MEMORY_SIZE)
        self.act_memo    = deque([],maxlen = MEMORY_SIZE)
        self.reward_memo = deque([],maxlen = MEMORY_SIZE)
        self.done_memo   = deque([],maxlen = MEMORY_SIZE)
        
        self.priority  = deque([],maxlen = MEMORY_SIZE)
        self.prob      = deque([],maxlen = MEMORY_SIZE)
        
    def memo_append(self, ob, act,reward, ob_next, done):
        if len(self.priority) == 0:
            self.priority.append(1)
        else:            
            self.priority.append(max(self.priority))
        # a_set_memory = sars(a) : [ob, (act), reward, ob_next, done]
        self.state_memo.append(ob)
        self.state_next_memo.append(ob_next)        
        self.act_memo.append(act)
        self.reward_memo.append(reward)
        self.done_memo.append(done)
        
        
    def memo_len(self):
        return len(self.state_memo)
        
    def prob_update(self):
        priority_alpha = np.power(np.array(self.priority),self.ALPHA)
        self.prob = priority_alpha/np.sum(priority_alpha)
        
    def priority_update(self, error, batch_index):
        priority_array = np.array(self.priority).astype('float') # somehow turn into int64
        priority_array[batch_index] = np.abs(error) + self.BASE
        self.priority = deque(priority_array.tolist(),maxlen = self.MEMORY_SIZE)
        
    def sample(self):
        
        self.prob_update()
        batch_index = np.random.choice(range(self.memo_len()),self.BATCH_SIZE,p=self.prob,replace=False)
        
        batch_state      = np.array(self.state_memo)[batch_index]
        batch_state_next = np.array(self.state_next_memo)[batch_index]
        batch_act        = np.array(self.act_memo)[batch_index]
        batch_reward     = np.array(self.reward_memo)[batch_index]
        batch_done       = np.array(self.done_memo)[batch_index]
        
        return batch_index, batch_state, batch_act, batch_reward, batch_state_next, batch_done

class Agent(PReplay,DDQN):
    def __init__(self, N_ACT,N_OB,                 \
                    ALPHA   = 0.5,                 \
                    BETA = 0.5,                    \
                    BASE =0.1,                     \
                    GAMMA   = 0.9,                 \
                    EPSILON = 1.0,                 \
                    MODEL_TRAIN_STEP = 4,                \
                    MODEL_UPDATE_STEP   = 200,    \
                    MEMORY_SAMPLE_START = 20,     \
                    LEARNING_RATE = 0.01,          \
                    MEMORY_SIZE  = 10_000,         \
                    BATCH_SIZE   = 64 ):

        PReplay.__init__(self, MEMORY_SIZE = MEMORY_SIZE, \
                        ALPHA = ALPHA, \
                        BETA = BETA, \
                        BASE = BASE,\
                        BATCH_SIZE = BATCH_SIZE)
        
        DDQN.__init__(self, N_ACT,N_OB,\
                     LEARNING_RATE = LEARNING_RATE)

        #self.N_ACT   = N_ACT
        self.GAMMA   = GAMMA
        self.EPSILON = EPSILON

        self.target_model = self.create_dqn()
        self.target_model.set_weights(self.model.get_weights())

        self.MODEL_UPDATE_STEP = MODEL_UPDATE_STEP
        self.MODEL_TRAIN_STEP  = MODEL_TRAIN_STEP
        
        self.UPDATE_STEP = 0
        self.TRAIN_STEP = 0
        
        self.MEMORY_SAMPLE_START = MEMORY_SAMPLE_START
        
    
    def get_q_value(self, state):
        # state: (1,84,84,4)
        return self.model.predict(state)
    
    
    def get_action(self,state): # get action with epsilon greedy
        # state modify for dm_wrapper
        
        q = self.get_q_value(state)
        if np.random.rand() < self.EPSILON:            
            return np.random.randint(self.N_ACT), np.amax(q)
        return np.argmax(q), np.amax(q)
    

    def train(self): 
        #if the momery len > 0.2 memory size
        if self.memo_len() < self.MEMORY_SAMPLE_START:
            return
        
        if self.TRAIN_STEP < self.MODEL_TRAIN_STEP:
            self.TRAIN_STEP+=1
            return
        
        self.TRAIN_STEP = 0
        batch_index, batch_state, batch_act, batch_reward, batch_state_next, batch_done = self.sample()
        
        # model for q now
        batch_q = self.model.predict(batch_state) # [32,4]
        # target_model for max q
        batch_q_next = self.target_model.predict(batch_state_next)
        
        
        batch_q_target = np.copy(batch_q)
        
        batch_max_act = np.argmax(batch_q,axis=1) # [32]
        
        batch_q_target[range(self.BATCH_SIZE),batch_act] = batch_reward + \
            (1-batch_done)*0.9*batch_q_next[range(self.BATCH_SIZE),batch_max_act]

        error = (batch_q_target - batch_q)[range(self.BATCH_SIZE),batch_act]
        
        self.priority_update(error,batch_index)
        # TODO: use weight to modify
        history = self.model.fit(batch_state,batch_q_target,batch_size = self.BATCH_SIZE, verbose = 0)
        
        self.UPDATE_STEP +=1
        return history.history
        
    def target_model_update(self):
        if self.UPDATE_STEP < self.MODEL_UPDATE_STEP:
            return
        self.UPDATE_STEP = 0
        self.target_model.set_weights(self.model.get_weights())
        
# %%
FRAME_END = args.FRAME_END
FRAME_T  = args.FRAME_T

ALPHA   = args.ALPHA
BETA    = args.BETA
BASE    = args.BASE

GAMMA   = args.GAMMA

EPSILON = args.EPSILON
EPSILON_END   = args.EPSILON_END

LEARNING_RATE = args.LEARNING_RATE

BATCH_SIZE = args.BATCH_SIZE

MEMORY_SIZE = args.MEMORY_SIZE
MEMORY_SAMPLE_START = args.MEMORY_SAMPLE_START

MODEL_TRAIN_STEP  = args.MODEL_TRAIN_STEP
MODEL_UPDATE_STEP = args.MODEL_UPDATE_STEP

# %%
env_name = "BreakoutNoFrameskip-v4"
env = make_env(env_name)
N_ACT = env.action_space.n
N_OB  = env.observation_space.shape # 84.84.4


agent = Agent(N_ACT,N_OB, \
              ALPHA = ALPHA, BETA = BETA, BASE = BASE,\
              GAMMA = GAMMA, EPSILON = EPSILON, \
              MODEL_TRAIN_STEP = MODEL_TRAIN_STEP,   \
              MODEL_UPDATE_STEP   = MODEL_UPDATE_STEP, \
              MEMORY_SAMPLE_START = MEMORY_SAMPLE_START, \
              LEARNING_RATE = LEARNING_RATE, \
              MEMORY_SIZE   = MEMORY_SIZE, \
              BATCH_SIZE    = BATCH_SIZE)

epsilon_decay = (EPSILON-EPSILON_END)/FRAME_T


history_summary = {
    'loss':[],
    'accuracy':[]
}
max_q_summary = []
reward_summary = []

# %%

ROOT_DIR = '../test_per_ddqn_break'
DIR = os.path.join(ROOT_DIR,datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
try:
    os.makedirs(DIR)
except:
    pass

DIR_GIF = os.path.join(DIR,'GIF')
try:
    os.mkdir(DIR_GIF)
except:
    pass

log_file = open(DIR+'/log.txt','a')
log_file.write("FRAME_END:{}, FRAME_T:{}, GAMMA:{}, EPSILON:{},  EPSILON_END:{}, LEARNING_RATE:{}\n".format(FRAME_END,FRAME_T,GAMMA,EPSILON,EPSILON_END,LEARNING_RATE))
log_file.write("BATCH_SIZE:{}, MEMORY_SIZE:{}, MODEL_UPDATE_STEP:{}, MEMORY_SAMPLE_START:{}\n".format(BATCH_SIZE,MEMORY_SIZE,MODEL_UPDATE_STEP,MEMORY_SAMPLE_START))
log_file.write("ALPHA:{}, BETA:{}, BASE:{}\n".format(ALPHA,BETA,BASE))
log_file.close()

frame_sum = 0
ep = 0
# %%
while(frame_sum<FRAME_END):

    
    log_file = open(DIR+'/log.txt','a')

    loss        = []
    accuracy    = []
    reward_list = []
    max_q_list  = []
    frame       = 0

    images = []
    ob = env.reset()
    #======================================================
    while(1):
        
        agent.EPSILON = max([EPSILON_END,agent.EPSILON-epsilon_decay])

        #env.render()
        images.append(env.render(mode='rgb_array'))

        act,max_q = agent.get_action(tf.expand_dims(ob.concatenate(),0))
        ob_next, reward, done, info = env.step(act)
        max_q_list.append(max_q)
        
        #ob_next_array = ob_next.concatenate()        
        agent.memo_append(ob.concatenate(),act,reward,ob_next.concatenate(),done)
        history = agent.train()
        agent.target_model_update()       
        
        ob = ob_next
        frame += 1
        
        reward_list.append(reward)
        if history:
            loss.append(history['loss'][0])
            accuracy.append(history['accuracy'][0])
            
        if done:            
            break
    ep += 1
    frame_sum += frame
    
    reward_summary.append(sum(reward_list))            
    max_q_summary.append(max(max_q_list))
    
    out = "\nEpoch {} \tframe: {} \tframe_sum:{} \tepsilon: {:0.5f} \tsum_rewards: {} \tmax_q:{:0.3f} \t".format(
             ep,frame,frame_sum,agent.EPSILON,reward_summary[-1],max_q_summary[-1])
    log_file.write(out)
    print(out,end=" ")

    if len(loss):
        history_summary['loss'].append(sum(loss)/len(loss))
        history_summary['accuracy'].append(sum(accuracy)/len(accuracy))
        
        out = "ave loss: {:.5f} \tave accuracy: {:.5f}\t".format(history_summary['loss'][-1],history_summary['accuracy'][-1])
        log_file.write(out)
        print(out,end=" ")        
        
    log_file.close()
    if reward_summary[-1]:
        gifname=str(ep)+'_r_'+str(reward_summary[-1])
        imageio.mimsave(os.path.join(DIR_GIF,gifname+'.gif'),images,fps=55)
        
# %%
#=============== Show the reward every ep
print('Show the reward every ep')
plt.figure()
plt.plot(reward_summary,label='max')
plt.savefig(DIR + '/rewards.png')

plt.figure()
plt.plot(history_summary['loss'],label='loss')
plt.plot(history_summary['accuracy'],label='accuracy')
plt.legend(loc=2)
plt.savefig(DIR + '/loss_accuracy.png')

plt.figure()
plt.plot(max_q_summary,label='max_q')
plt.savefig(DIR + '/max_q.png')
   
DIR_MODEL = os.path.join(DIR,'Model')
os.mkdir(DIR_MODEL)
agent.model.save(DIR_MODEL)

# %%
#==================================================
print('Test the final round')
# observe the final run

log_file = open(DIR+'/log.txt','a')

reward_list = []
images = []
frame  = 0
env.was_real_done = True
ob = env.reset()
while(1):
    #env.render()
    #pdb.set_trace()

    images.append(env.render(mode='rgb_array'))

    act = np.argmax(agent.get_q_value(tf.expand_dims(ob.concatenate(),0)))

    ob_next,reward,done,info = env.step(act)
    
    reward_list.append(reward)
    frame +=1
    ob = ob_next
    
    if done:        
        break
out = '\nFinal:\tframe: {} sum_rewards: {}\t'.format(frame,sum(reward_list))
log_file.write(out)
print(out)

log_file.close()       
gifname='Final_r_'+str(sum(reward_list))
imageio.mimsave(os.path.join(DIR_GIF,gifname+'.gif'),images,fps=55)
env.close()
print("Done")