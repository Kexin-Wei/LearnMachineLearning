import os
import imageio
import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import deque,namedtuple
from dm_wrapper import make_env
"""### Args"""

FRAME_END = 5e5

GAMMA  = 0.99

EPSILON      = 1.0
EPSILON_END  = 0.02
EPSILON_DECAY = 0.999988

LEARNING_RATE = 0.0001

BATCH_SIZE  = 32

MEMORY_SIZE = 40_000
MEMORY_SAMPLE_START = 1_000

MODEL_UPDATE_STEP =  40

RENDER = False
GIF_MAKE = False

ENV_NAME = "PongNoFrameskip-v4"

"""### DuelingDQN"""

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
        
        x1 = self.value_fc2(x1)
        x2 = self.advantage_fc2(x2)
        
        # expand and repeat https://zhuanlan.zhihu.com/p/58109107
        
        x = x2-x2.mean(dim=1,keepdim=True)+x1
        return x

"""### DQN"""

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

"""### Priorited Replay"""

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
        batch_act        = np.array(self.act_memo)[batch_index].astype(int)
        batch_reward     = np.array(self.reward_memo)[batch_index]
        batch_done       = np.array(self.done_memo)[batch_index]
        
        return batch_index,batch_state, batch_act, batch_reward, batch_state_next, batch_done

"""### Replay"""
memory = namedtuple('Experience', ['state', 'action', 'reward', 'next_state',  'done'])

class Replay:
    def __init__(self,MEMORY_SIZE = 5000, BATCH_SIZE = 32):
        self.BATCH_SIZE  = BATCH_SIZE
        self.MEMORY_SIZE = MEMORY_SIZE
        
        self.memo = deque([],maxlen = MEMORY_SIZE)
        
    def memo_append(self, ob, act,reward,   ob_next, done,):
        # a_set_memory = sars(a) : [ob, (act), reward, ob_next, done]
        # ob modify for dm_wrapper                                    
        self.memo.append(memory(ob,act,reward, ob_next ,done))
        
    def memo_len(self):
        return len(self.memo)
        
    
    def sample(self):
        batch_index = np.random.choice(range(self.memo_len()),self.BATCH_SIZE,replace=False)
        
        batch_state, batch_act, batch_reward, batch_state_next, batch_done= zip(*[self.memo[idx] for idx in batch_index])
        return np.array(batch_index), np.array(batch_state), \
            np.array(batch_act).astype(int), np.array(batch_reward),  np.array(batch_state_next), np.array(batch_done)
            
"""## Main"""

# %%


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Running on GPU")
else:
    DEVICE = torch.device('cpu')
    print("Running on CPU")

env = make_env(ENV_NAME)
N_ACT = env.action_space.n
N_OB  = env.observation_space.shape # 84.84.4


FILENAME = os.path.splitext(os.path.basename(__file__))[0]
FILENAME = "dqn"
ROOT = f"../test_{FILENAME}_{ENV_NAME}"
DIR = os.path.join(ROOT, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
try:
    os.makedirs(DIR)
except:
    print(f"Failed to open folder {DIR}")

DIR_GIF = os.path.join(DIR,'GIF')
try:
    os.makedirs(DIR_GIF)
except:
    print(f"Failed to open folder {DIR_GIF}")
    

# log_file = Logfile(DIR+'/log.txt')
print(f"FRAME_END:{FRAME_END}\n"   
      f"EPSILON:{EPSILON}, EPSILON_END:{EPSILON_END},EPSILON_DECAY:{EPSILON_DECAY}\n"   
      f"GAMMA: {GAMMA},  LEARNING_RATE:{LEARNING_RATE} \n"  
      f"MEMORY_SIZE:{MEMORY_SIZE} ,MEMORY_SAMPLE_START:{MEMORY_SAMPLE_START}\n" 
      f"MODEL_UPDATE_STEP:{MODEL_UPDATE_STEP}\n"  
      f"BATCH_SIZE:{BATCH_SIZE}\n")

ReplayBuffer = Replay(MEMORY_SIZE = MEMORY_SIZE,BATCH_SIZE = BATCH_SIZE)

model = DQN(N_OB[2],N_ACT).to(DEVICE)
target_model = DQN(N_OB[2],N_ACT).to(DEVICE)
target_model.load_state_dict(model.state_dict())
        
optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)
loss_func = nn.MSELoss()
  
    
def get_action(state): # get action with epsilon greedy
    # state modify for dm_wrapper
    with torch.no_grad():
        q = model(state)
    if np.random.rand() <= EPSILON:            
        return np.random.randint(N_ACT)
    return torch.argmax(q)

    
def replay_sample_tensor():   
    batch_index, batch_state, batch_act, batch_reward, batch_state_next, batch_done = ReplayBuffer.sample()

    batch_state      = torch.Tensor(batch_state).permute(0,3,1,2).to(DEVICE)
    batch_state_next = torch.Tensor(batch_state_next).permute(0,3,1,2).to(DEVICE)
    
    
    batch_act    = torch.Tensor(batch_act).type(torch.LongTensor).to(DEVICE)
    batch_reward = torch.Tensor(batch_reward).to(DEVICE)
    batch_done   = torch.Tensor(batch_done).to(DEVICE)
    return batch_index,batch_state, batch_act, batch_reward, batch_state_next, batch_done

# %%               
reward_sum = []
frame_sum = 0
ep = 0

TEST = False
UPDATE_STEP = 0
# %%
while(1):

  reward_list = 0
  step = 0
  if GIF_MAKE: images = []
  ob = env.reset()

  # print(model.fc2.bias)
  # print(target_model.fc2.bias)
  #======================================================
  while(1):
      if TEST:
          EPSILON = 0
      else:
          EPSILON = max([EPSILON_END,EPSILON*EPSILON_DECAY])
          
      if RENDER:
          env.render()
      if GIF_MAKE:
          images.append(env.render(mode='rgb_array'))

      state = torch.Tensor(ob.concatenate()).view(-1,N_OB[2],N_OB[0],N_OB[1]).to(DEVICE)
      act = get_action(state)
      ob_next, reward, done, info = env.step(act)

      #ob_next_array = ob_next.concatenate()        
      ReplayBuffer.memo_append(ob.concatenate(),act,reward,ob_next.concatenate(),done)
      
      # batch train==============================
      if ReplayBuffer.memo_len() >= MEMORY_SAMPLE_START:
          UPDATE_STEP+=1
          batch_index,batch_state, batch_act, batch_reward, batch_state_next, batch_done = replay_sample_tensor()
          
          range_index = torch.arange(0,BATCH_SIZE,device=DEVICE)
          batch_act = batch_act.type(torch.LongTensor).to(DEVICE)
          predict_q = model(batch_state)[range_index,batch_act] # [32,4] batch_q = model(batch_state)
          
          # target_model for max q
          with torch.no_grad():
              target_q = batch_reward + \
                          (1-batch_done)*GAMMA*torch.amax(target_model(batch_state_next),dim=1)
          # pytorch sgd
          del batch_state, batch_act, batch_reward, batch_state_next, batch_done 

          #ReplayBuffer.priority_update((target_q-predict_q).cpu().detach().numpy(),batch_index)
          optimizer.zero_grad()
          loss = loss_func(predict_q,target_q)    
          #loss = loss.clamp(-1,1)
          loss.backward()
          optimizer.step()                

      # copy parameters==============================
      if UPDATE_STEP >= MODEL_UPDATE_STEP:
        UPDATE_STEP=0
          # print("\nModel Bias:",model.fc2.bias.data)
        target_model.load_state_dict(model.state_dict()) 
      
      ob = ob_next
      step += 1
      frame_sum +=1
      reward_list += reward
          
      if done:            
          break
  #==================================    
  ep += 1

  reward_sum.append(reward_list)

  print(f"\nEpoch: {ep} \tstep: {step}" 
        f"\tframe_sum: {frame_sum}"
        f"\tepsilon: {EPSILON:.3f}"
        f"\tmean_rewards: {np.mean(reward_sum[-100:]):.3f}"
        f"\tsum_rewards: {reward_sum[-1]:.3f}",end="")


  if GIF_MAKE and reward_sum[-1]: # last reward > 0
      gifname=f"final_r_{reward_sum[-1]}.gif" if TEST else f"{ep}_r_{reward_sum[-1]}.gif"
      imageio.mimsave(os.path.join(DIR_GIF,gifname),images,fps=40)

  if TEST:        
      break

  if frame_sum>FRAME_END or reward_sum[-1]>20:
      TEST = True
      GIF_MAKE = True
    
        # train end, rund
#=============== Show the reward every ep
print('Show the reward every ep')
plt.figure()
plt.plot(reward_sum)
plt.savefig(DIR + '/rewards.png')

torch.save(model.state_dict(),os.path.join(DIR,'Model.pt'))
print("Done")

