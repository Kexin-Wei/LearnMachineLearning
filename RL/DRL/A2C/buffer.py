from collections import deque,namedtuple
import numpy as np

memory = namedtuple('Experience', ['state', 'action', 'reward','next_state','done'])
class Replay:
    def __init__(self,MEMORY_SIZE = 5000, BATCH_SIZE = 32):
        self.BATCH_SIZE  = BATCH_SIZE
        self.MEMORY_SIZE = MEMORY_SIZE
        
        self.memo = deque([],maxlen = MEMORY_SIZE)
        
    def append(self, ob, act,reward, ob_next, done):
        # a_set_memory = sars(a) : [ob, (act), reward, ob_next, done]
        # ob modify for dm_wrapper                                    
        self.memo.append(memory(ob,act,reward,ob_next ,done))
        
    def __len__(self):
        return len(self.memo)
            
    def sample(self):
        batch_index = np.random.choice(range(len(self.memo)),self.BATCH_SIZE,replace=False)
        
        batch_state, batch_act, batch_reward, batch_state_next , batch_done = zip(*[self.memo[idx] for idx in batch_index])
        return np.array(batch_index), np.array(batch_state), \
            np.array(batch_act).astype(int), np.array(batch_reward), np.array(batch_state_next),  np.array(batch_done)
            
class PER_Replay:
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
        
        self.memo = deque([],maxlen = MEMORY_SIZE)
        
        self.priority  = deque([],maxlen = MEMORY_SIZE)
        self.prob      = deque([],maxlen = MEMORY_SIZE)
        
    def append(self, ob, act,reward, ob_next, done):
        if len(self.priority) == 0:
            self.priority.append(1.0)
        else:            
            self.priority.append(max(self.priority))
        # a_set_memory = sars(a) : [ob, (act), reward, ob_next, done]
        self.memo.append(memory(ob,act,reward,ob_next ,done))
        
    def __len__(self):
        return len(self.memo)
        
    def prob_update(self):
        """priority_alpha = np.power(np.array(self.priority),self.ALPHA)
        self.prob = priority_alpha/np.sum(priority_alpha)"""
        pass
        
    def priority_update(self, error,batch_index):
        """priority_array = np.array(self.priority).astype('float') # somehow turn into int64
        priority_array[batch_index] = np.abs(error) + self.BASE
        self.priority = deque(priority_array.tolist(),maxlen = self.MEMORY_SIZE)"""
        pass
        
    def sample(self):
        
        # self.prob_update()
        batch_index = np.random.choice(range(self.memo_len()),self.BATCH_SIZE,p=self.prob,replace=False)
        
        batch_state, batch_act, batch_reward, batch_state_next , batch_done = zip(*[self.memo[idx] for idx in batch_index])
        return np.array(batch_index), np.array(batch_state), \
            np.array(batch_act).astype(int), np.array(batch_reward), np.array(batch_state_next),  np.array(batch_done)