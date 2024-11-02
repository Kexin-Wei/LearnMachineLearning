#============= Class Agent
import tensorflow as tf
import numpy as np
from collections import deque
import random
class DQN_Agent:
    def __init__(self, N_ACT, N_OB, \
                 MEMORY_SIZE = 2000,BATCH_SIZE = 32, WRAPPER_SIZE = 4, \
                 EPSILON = 0.3, GAMMA=0.9, EPSILON_DC = 0.9997):
        self.N_ACT   = N_ACT
        self.N_OB    = N_OB
        
        self.EPSILON = EPSILON
        self.GAMMA   = GAMMA
        self.EPSILON_DC = EPSILON_DC
        
        self.BATCH_SIZE = BATCH_SIZE
        self.MEMORY_SIZE = MEMORY_SIZE
        
        self.model  = self.create_cnn()
        
        self.target_model = self.create_cnn()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen = MEMORY_SIZE)
        
    def create_cnn(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(8,3, padding = 'same', activation = 'relu', input_shape = self.N_OB),
            tf.keras.layers.Conv2D(8,3, padding = 'same', activation = 'relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation = 'relu'),
            tf.keras.layers.Dense(self.N_ACT, activation = 'linear')
        ])
        
        model.compile(
            loss = 'huber_loss',
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics   = ['accuracy']
        )
        return model
    
    #==== get q-value
    def get_q(self, ob):
        return self.model.predict(ob.reshape(-1,self.N_OB[0],self.N_OB[1],self.N_OB[2])/255.0)
    
    #==== act = take_action()
    def take_action(self,ob): 
        if np.random.rand() < self.EPSILON:
            return np.random.randint(self.N_ACT)
        q_value = self.get_q(ob)
        return np.argmax(q_value)
    
    
    #===== self.replay_memory <- add(ob,act,reward,ob_next)
    def memorize(self,a_set_memory): 
        # a_set_memory = sars(a) : [ob, act, reward, ob_next, done]
        self.replay_memory.append(a_set_memory)
    
    #==== batch train 
    def train(self):       
        
        batch_memory = random.sample(self.replay_memory,self.BATCH_SIZE)
        
        batch_ob  = np.array([ a_set_memory[0] for a_set_memory in batch_memory])/255
        
        batch_ob_next = np.array([ a_set_memory[3] for a_set_memory in batch_memory])/255        
        batch_q_next  = self.target_model.predict(batch_ob_next)
        #set_trace()
        batch_q_new = []
        # loss = (reward+ q'-q)^2/batch_size
        for index,(ob, act, reward, ob_next, done) in enumerate(batch_memory):
            if not done:
                q_next_max = np.max(batch_q_next[index])
                q_new    = reward + self.GAMMA * q_next_max
            else:
                q_new    = reward 
            batch_q_new.append(q_new)
             
        self.model.fit(batch_ob,np.array(batch_q_new),batch_size = self.BATCH_SIZE, verbose = 0)
        
    
    #==== target_model <- model
    def target_model_update(self):
        self.target_model.set_weights(self.model.get_weights())
        