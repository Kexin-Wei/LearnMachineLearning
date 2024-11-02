# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf
import numpy as np
import random

from collections import deque
from dm_wrapper import make_env


# %%
class DDQN:
    # Reference:
    # https://github.com/EvolvedSquid/tutorials/blob/master/dqn/train_dqn.ipynb
    # TODO: modify for importance sampling
    def __init__(self, N_ACT,LEARNING_RATE = 0.01):
        self.INPUT_SIZE = (84,84,4)
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
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE),
            loss = tf.keras.losses.Huber(),
            metrics = ['accuracy']
        )
        return model

# %%
# change replay to prioried replay

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
        priority_array = np.array(self.priority)
        priority_array[batch_index] = np.abs(error) + self.BASE
        self.priority = deque(priority_array.tolist(),maxlen = self.MEMORY_SIZE)
        
    def sample(self):
        self.prob_update()
        
        batch_index = np.random.choice(range(self.memo_len()),p=self.prob,replace=False)
        
        batch_state      = np.array(self.state_memo)[batch_index]
        batch_state_next = np.array(self.state_next_memo)[batch_index]
        batch_act        = np.array(self.act_memo)[batch_index]
        batch_reward     = np.array(self.reward_memo)[batch_index]
        batch_done       = np.array(self.done_memo)[batch_index]
        
        return batch_index, batch_state, batch_act, batch_reward, batch_state_next, batch_done
# %%
# TODO: Change Agent class 
class Agent(PReplay,DDQN):
    def __init__(self, N_ACT,N_OB,                 \
                    GAMMA   = 0.9,                 \
                    EPSILON = 1.0,                 \
                    MODEL_UPDATE_STEP   = 200,    \
                    MEMORY_SAMPLE_START = 20,     \
                    LEARNING_RATE = 0.01,          \
                    MEMORY_SIZE  = 10_000,         \
                    BATCH_SIZE   = 64 ):

        PReplay.__init__(self, MEMORY_SIZE = MEMORY_SIZE, \
                        BATCH_SIZE = BATCH_SIZE)
        
        DDQN.__init__(self, N_ACT,N_OB,\
                     LEARNING_RATE = LEARNING_RATE)

        #self.N_ACT   = N_ACT
        self.GAMMA   = GAMMA
        self.EPSILON = EPSILON

        self.target_model = self.create_dqn()
        self.target_model.set_weights(self.model.get_weights())

        self.MODEL_UPDATE_STEP = MODEL_UPDATE_STEP
        self.STEP = 0
        
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
        batch_index, batch_state, batch_act, batch_reward, batch_state_next, batch_done = self.sample()
        
        # model for q now
        batch_q = self.model.predict(batch_state) # [32,4]
        # target_model for max q
        batch_q_next = self.target_model.predict(batch_state_next)
        
        
        batch_q_target = np.copy(batch_q)
        
        batch_max_act = np.argmax(batch_q,axis=1) # [32]
        
        batch_q_target[range(self.BATCH_SIZE),batch_act] = batch_reward + \
            (1-batch_done)*0.9*batch_q_next[range(self.BATCH_SIZE),batch_max_act]

        error = (batch_q_target - batch_q_target)[range(self.BATCH_SIZE),batch_act]
        
        self.priority_update(error,batch_index)
        # TODO: use weight to modify
        history = self.model.fit(batch_state,batch_q_target,batch_size = self.BATCH_SIZE, verbose = 0)
        
        self.STEP +=1
        return history.history
        
    def target_model_update(self):
        if self.STEP < self.MODEL_UPDATE_STEP:
            return
        self.STEP = 0
        self.target_model.set_weights(self.model.get_weights())
#%%

# %% Test for DDQN

input_layer = tf.keras.layers.Input(shape=(84,84,4))
cnn1 = tf.keras.layers.Conv2D(32,8,strides=4,activation='relu')(input_layer)
cnn2 = tf.keras.layers.Conv2D(64,4,strides=2,activation='relu')(cnn1)
cnn3 = tf.keras.layers.Conv2D(64,3,strides=1,activation='relu')(cnn2)
flatten = tf.keras.layers.Flatten()(cnn3)
value = tf.keras.layers.Dense(1)(flatten)
advantage = tf.keras.layers.Dense(4)(flatten)

reduce_mean = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x,axis=1,keepdims=True)) # custom a type of layer
q_output = tf.keras.layers.Add()([value,\
                                tf.keras.layers.Subtract()([advantage,reduce_mean(advantage)])])
model = tf.keras.Model(input_layer,q_output)

model.summary() 

# %%
env = make_env('BreakoutNoFrameskip-v4')
ob = env.reset()
ob_numpy = ob.concatenate()
print(ob_numpy.shape)
print(model.predict(tf.expand_dims(ob_numpy,0)))

# %% [markdown]
### tf.expand_dims()

#%%
print(tf.expand_dims(ob_numpy,0).shape)
# %% [markdown]
### tf.split()

# %%
x=np.random.randint(5,size=[3,3,3])
print(x.shape)
y=tf.split(x,3,0)
print(len(y))
print(y[0].shape)
print(y[1].shape)
print(y[2].shape)
# %% [markdown]
### tf.math.reduce_mean()

# %%
x=np.random.randint(6,size=[5,1])
print(x)
y=tf.reduce_mean(x,keepdims= True)
print(y)
y_x= tf.keras.Sub
# %%
import random
l= 10
batch_index = random.sample(range(l),3)
print(batch_index)
# %%
from collections import deque
import numpy as np
state=deque(range(l),maxlen = 10)
print(state)
print(np.array(state)[batch_index])

# %%
