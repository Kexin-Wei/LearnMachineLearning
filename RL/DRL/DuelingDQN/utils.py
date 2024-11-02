#%%
from collections import deque,namedtuple
import numpy as np

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
    
# %%        
# how to get 7*7*64
"""class DDQN(nn.Module):
    def __init__(self,IN_CHANNEL,N_ACT):
        super().__init__()
        
        self.conv1 = nn.Conv2d(IN_CHANNEL,32,8,stride=4)
        self.conv2 = nn.Conv2d(32,64,4,stride=2)
        self.conv3 = nn.Conv2d(64,64,3,stride=1)
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x
ddqn = DDQN(N_OB[2],N_ACT).to(device)
optimizer = optim.Adam(ddqn.parameters(),lr = 0.001)
loss_function = nn.MSELoss()
ob = env.reset()
y = ddqn(torch.Tensor(ob).view(-1,N_OB[2],N_OB[0],N_OB[1]))
print(y.shape)"""

class DDQN(nn.Module):
    def __init__(self,IN_CHANNEL,N_ACT):
        super().__init__()
        
        self.conv1 = nn.Conv2d(IN_CHANNEL,32,8,stride=4)
        self.conv2 = nn.Conv2d(32,64,4,stride=2)
        self.conv3 = nn.Conv2d(64,64,3,stride=1)
        
        self.value_fc1 = nn.Linear(7*7*64,512)
        self.value_fc2 = nn.Linear(512,1)
        
        self.advantage_fc1 = nn.Linear(7*7*64,512)
        self.advantage_fc2 = nn.Linear(512,N_ACT)
        
        #self.qvalue = nn.Linear(N_ACT)
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.reshape(x.shape[0],-1)
        x1 = F.relu(self.value_fc1(x))
        x2 = F.relu(self.advantage_fc1(x))
        
        # x1 = F.relu(self.value_fc2(x1))
        # x2 = F.relu(self.advantage_fc2(x2))
        
        x1 = self.value_fc2(x1)
        x2 = self.advantage_fc2(x2)
        
        # expand and repeat https://zhuanlan.zhihu.com/p/58109107
        
        x = x2-x2.mean(dim=1,keepdim=True)+x1
        return x
    
class DQN_FC(nn.Module):
    def __init__(self,N_ACT,*args):
        super().__init__()
        if args:
            fc_num = args[0]
            assert len(fc_num) == 2
            fc1, fc2 = fc_num[0],fc_num[1]
        else:
            fc1, fc2 = 256,256
        self.fc1 = nn.Linear(84*84*4,fc1)
        self.fc2 = nn.Linear(fc1,fc2)
        self.out = nn.Linear(fc1,N_ACT)
        
    def forward(self,x):
        x = F.relu(self.fc1(x.reshape(-1,84*84*4)))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x 
    
class DQN(nn.Module):
    def __init__(self,IN_CHANNEL,N_ACT):
        super().__init__()
        
        self.conv1 = nn.Conv2d(IN_CHANNEL,32,8,stride=4)
        self.conv2 = nn.Conv2d(32,64,4,stride=2)
        self.conv3 = nn.Conv2d(64,64,3,stride=1)
        
        self.fc1 = nn.Linear(7*7*64,512)
        self.fc2 = nn.Linear(512,N_ACT)
        
        #self.qvalue = nn.Linear(N_ACT)
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.reshape(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# %%
memory = namedtuple('Experience', ['state', 'action', 'reward','next_state','done'])
class Replay:
    def __init__(self,MEMORY_SIZE = 5000, BATCH_SIZE = 32):
        self.BATCH_SIZE  = BATCH_SIZE
        self.MEMORY_SIZE = MEMORY_SIZE
        
        self.memo = deque([],maxlen = MEMORY_SIZE)
        
    def memo_append(self, ob, act,reward, ob_next, done):
        # a_set_memory = sars(a) : [ob, (act), reward, ob_next, done]
        # ob modify for dm_wrapper                                    
        self.memo.append(memory(ob,act,reward,ob_next ,done))
        
    def memo_len(self):
        return len(self.memo)
            
    def sample(self):
        batch_index = np.random.choice(range(self.memo_len()),self.BATCH_SIZE,replace=False)
        
        batch_state, batch_act, batch_reward, batch_state_next , batch_done = zip(*[self.memo[idx] for idx in batch_index])
        return np.array(batch_index), np.array(batch_state), \
            np.array(batch_act).astype(int), np.array(batch_reward), np.array(batch_state_next),  np.array(batch_done)
    
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
        
        self.memo = deque([],maxlen = MEMORY_SIZE)
        
        self.priority  = deque([],maxlen = MEMORY_SIZE)
        self.prob      = deque([],maxlen = MEMORY_SIZE)
        
    def memo_append(self, ob, act,reward, ob_next, done):
        if len(self.priority) == 0:
            self.priority.append(1.0)
        else:            
            self.priority.append(max(self.priority))
        # a_set_memory = sars(a) : [ob, (act), reward, ob_next, done]
        self.memo.append(memory(ob,act,reward,ob_next ,done))
        
    def memo_len(self):
        return len(self.memo)
        
    def prob_update(self):
        priority_alpha = np.power(np.array(self.priority),self.ALPHA)
        self.prob = priority_alpha/np.sum(priority_alpha)
        
    def priority_update(self, error,batch_index):
        priority_array = np.array(self.priority).astype('float') # somehow turn into int64
        priority_array[batch_index] = np.abs(error) + self.BASE
        self.priority = deque(priority_array.tolist(),maxlen = self.MEMORY_SIZE)
        
    def sample(self):
        
        self.prob_update()
        batch_index = np.random.choice(range(self.memo_len()),self.BATCH_SIZE,p=self.prob,replace=False)
        
        batch_state, batch_act, batch_reward, batch_state_next , batch_done = zip(*[self.memo[idx] for idx in batch_index])
        return np.array(batch_index), np.array(batch_state), \
            np.array(batch_act).astype(int), np.array(batch_reward), np.array(batch_state_next),  np.array(batch_done)
# %%
class Logfile:
    def __init__(self, DIR):
        self.DIR = DIR
        self.log_file = open(DIR,'a')    
        
    def open(self):
        self.log_file = open(self.DIR,'a')
        
    def write(self,string):
        print(string,end="")        
        self.log_file.write(string)
        
    def close(self):
        self.log_file.close()
# %%
