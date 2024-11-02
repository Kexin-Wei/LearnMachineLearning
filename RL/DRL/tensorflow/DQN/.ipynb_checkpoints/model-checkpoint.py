import tensorflow as tf
import numpy as np
import random
import cv2

from collections import deque

# ### ObWrapper Class

# make sure import deque from collections 
# and opencv(cv2)
# and numpy as np
class ObWrapper:
    def __init__(self, WRAPPER_SIZE = 4 ):
        self.WRAPPER_SIZE = WRAPPER_SIZE
        self.s = deque([],maxlen = WRAPPER_SIZE) #wrapper how many frame together
        
    def __call__(self,ob):
        gray = cv2.cvtColor(ob,cv2.COLOR_BGR2GRAY)
        self.s.append(cv2.resize(gray,(84,110),cv2.INTER_NEAREST)[16:100,:])

    def __len__(self):
        return len(self.s)
    
    def packup(self):
        if len(self.s) < self.WRAPPER_SIZE:
            return print("Wrapper too small, unpackable")
        a = np.array([self.s[i] for i in range(self.WRAPPER_SIZE)])
        b = np.transpose(a,(1,2,0))  # or b = np.einsum('ijk->jki',a)
        s1,s2 = self.s[0].shape
        return b.reshape(-1,s1,s2,self.WRAPPER_SIZE)


# make sure import random
# and deque from collections
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
        
        self.INPUT_SIZE  = self.input_size([84,84], WRAPPER_SIZE)
        self.OUTPUT_SIZE = N_ACT
        self.LEARNING_RATE = LEARNING_RATE
        self.model = self.create_cnn()
        
    def input_size(self,N_OB, WRAPPER_SIZE):
        return (N_OB[0],N_OB[1],WRAPPER_SIZE)
    
    def create_cnn(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16,8,strides = 2,input_shape =self.INPUT_SIZE,\
                                   activation = 'relu'),
            tf.keras.layers.Conv2D(32,4,strides = 2, activation = 'relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256,               activation = 'relu'),
            tf.keras.layers.Dense(self.OUTPUT_SIZE, activation = 'linear'),
        ])
        model.compile(
            loss = 'mse',
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE),
            metrics   = ['accuracy']
        )
        return model


# ### Agent
class Agent(Replay,CNN):
    def __init__(self, N_ACT,N_OB,                 \
                    GAMMA   = 0.9,                 \
                    EPSILON = 0.3,                 \
                    EPSILON_DECAY = 0.997,         \
                    MODEL_UPDATE_STEP   = 200,    \
                    MEMORY_SAMPLE_START = 0.2,     \
                    LEARNING_RATE = 0.01,          \
                    MEMORY_SIZE  = 10_000,         \
                    BATCH_SIZE   = 64,             \
                    WRAPPER_SIZE = 4 ):

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

        return self.model.predict(state.packup())
    
    
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


