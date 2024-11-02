# test for priority replay buffer
#%%
from DuelingDQN.duelingdqn import MEMORY_SIZE
import numpy as np
from collections import deque

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
        self.priority = deque(priority_array.tolist(),maxlen = MEMORY_SIZE)
        
    def sample(self):
        self.prob_update()
        
        batch_index = np.random.choice(range(self.memo_len()),p=self.prob,replace=False)
        
        batch_state      = np.array(self.state_memo)[batch_index]
        batch_state_next = np.array(self.state_next_memo)[batch_index]
        batch_act        = np.array(self.act_memo)[batch_index]
        batch_reward     = np.array(self.reward_memo)[batch_index]
        batch_done       = np.array(self.done_memo)[batch_index]
        
        return batch_state, batch_act, batch_reward, batch_state_next, batch_done
    
#%%
import numpy as np
l = 10
p = np.array(range(10))
p = p/sum(p)
print(p)
# %%
np.random.choice(range(l),3,p=p, replace=False)
# %%
from collections import deque
c=deque(range(l),maxlen=l)
print(np.array(c))
# %%
print(np.power(c,0.5))
# %%
prio = range(l)
prio_array = np.array(prio).astype('float')
error = np.random.randint(5,size=3)-2.5
index = np.random.choice(range(l),3,replace=False)
print(error)
print(index)
# %%
base = 0.1
prio_array[index] = np.abs(error) + base 
print(prio_array)
print(deque(prio_array.tolist(),maxlen=l))
# %%
