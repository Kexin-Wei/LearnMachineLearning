# %%
import os
import imageio
import datetime
import random

import matplotlib.pyplot as plt
import numpy as np
import math

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dm_wrapper import make_env
from utils import DDQN, PReplay, Logfile,DQN, Replay, DQN_FC

# %%        
FRAME_END = 3e6

GAMMA  = 0.99

EPSILON      = 1.0
EPSILON_END  = 0.1
EPSILON_DECAY = 0.9999977

LEARNING_RATE = 0.0001

BATCH_SIZE  = 32

MEMORY_SIZE = 1_000_000
MEMORY_SAMPLE_START = 5_000

MODEL_UPDATE_STEP =  10_000
TRAIN_SKIP_STEP   =  4

ENV_NAME = "PongNoFrameskip-v4"
ENV_NAME = "PongDet-v4"
# 
RENDER = True
CLIP_FLAG = False
GIF_MAKE  = False
DDQN_FLAG = False # False for dqn
PER_FLAG  = False # False for normal replay
DOUBLE_FLAG = False # True for double q learning
FC_FLAG   = True # True for fully connected network
fc_num    = [256,256]
# %%
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Running on GPU")
else:
    DEVICE = torch.device('cpu')
    print("Running on CPU")
# %%
env = make_env(ENV_NAME)
N_ACT = env.action_space.n
N_OB  = env.observation_space.shape # 84.84.4

# %%
FILENAME = os.path.splitext(os.path.basename(__file__))[0]
if DOUBLE_FLAG:
    FILENAME='double_'+FILENAME
if DDQN_FLAG:
    FILENAME="dueling_"+FILENAME
if PER_FLAG:
    FILENAME='per_'+FILENAME
if FC_FLAG:
    FILENAME='fc_'+FILENAME

ROOT = f"test_{FILENAME}_{ENV_NAME}"
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
    
# %%
log_file = Logfile(DIR+'/log.txt')
log_file.write(f"FRAME_END:{FRAME_END}\n"   
               f"EPSILON:{EPSILON}, EPSILON_END:{EPSILON_END}, EPSILON_DECAY:{EPSILON_DECAY}\n"   
               f"GAMMA: {GAMMA},  LEARNING_RATE:{LEARNING_RATE} \n"  
               f"MEMORY_SIZE:{MEMORY_SIZE} ,MEMORY_SAMPLE_START:{MEMORY_SAMPLE_START}\n" 
               f"MODEL_UPDATE_STEP:{MODEL_UPDATE_STEP}, TRAIN_SKIP_STEP:{TRAIN_SKIP_STEP}\n"  
               f"BATCH_SIZE:{BATCH_SIZE}\n")
log_file.close()
# %%
if PER_FLAG:
    ReplayBuffer = PReplay(MEMORY_SIZE = MEMORY_SIZE,BATCH_SIZE = BATCH_SIZE)
else:
    ReplayBuffer = Replay(MEMORY_SIZE = MEMORY_SIZE,BATCH_SIZE = BATCH_SIZE)
    
    

if DDQN_FLAG:
    model = DDQN(N_OB[2],N_ACT).to(DEVICE)
    target_model = DDQN(N_OB[2],N_ACT).to(DEVICE)

elif FC_FLAG:
    model = DQN_FC(N_ACT,fc_num).to(DEVICE)
    target_model = DQN_FC(N_ACT,fc_num).to(DEVICE)
else:
    model = DQN(N_OB[2],N_ACT).to(DEVICE)
    target_model = DQN(N_OB[2],N_ACT).to(DEVICE)
# %%  
target_model.load_state_dict(model.state_dict())        
optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)
loss_func = nn.MSELoss()
  
    
def get_action(state): # get action with epsilon greedy
    # state modify for dm_wrapper
    #with torch.no_grad():
    q = model(state)
    if np.random.rand() <= EPSILON:            
        return np.random.randint(N_ACT), torch.amax(q,dim=1).item()
    return torch.argmax(q), torch.amax(q,dim=1).item()

    
def replay_sample_tensor():   
    batch_index, batch_state, batch_act, batch_reward, batch_state_next, batch_done = ReplayBuffer.sample()

    batch_state      = torch.Tensor(batch_state).permute(0,3,1,2).to(DEVICE)
    batch_state_next = torch.Tensor(batch_state_next).permute(0,3,1,2).to(DEVICE)
    
    
    batch_act    = torch.Tensor(batch_act).type(torch.LongTensor).to(DEVICE)
    batch_reward = torch.Tensor(batch_reward).to(DEVICE)
    batch_done   = torch.Tensor(batch_done).to(DEVICE)
    return batch_index,batch_state, batch_act, batch_reward, batch_state_next, batch_done
               
# %%
loss_sum   = []
max_q_max  = []
reward_ave = []
reward_sum = []
frame_sum = 0
ep = 0

TEST = False
SKIP_STEP = 0 
UPDATE_STEP = 0
# %%
while(1):
    
    log_file.open()

    loss_list   = []
    reward_list = []
    max_q_list  = []
    step        = 0
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

        state = torch.Tensor(ob.cat()).view(-1,N_OB[2],N_OB[0],N_OB[1]).to(DEVICE)
        act, max_q = get_action(state)
        ob_next, reward, done, info = env.step(act)

        #ob_next_array = ob_next.concatenate()        
        ReplayBuffer.memo_append(ob.cat(),act,reward,ob_next.cat(),done)
        
        # batch train==============================
        if ReplayBuffer.memo_len() >= MEMORY_SAMPLE_START:
            if SKIP_STEP >= TRAIN_SKIP_STEP:
                SKIP_STEP = 0
                
                batch_index,batch_state, batch_act, batch_reward, batch_state_next, batch_done = replay_sample_tensor()
                
                range_index = torch.arange(0,BATCH_SIZE,device=DEVICE)
                batch_act = batch_act.type(torch.LongTensor).to(DEVICE)
                # model for q now
                predict_q = model(batch_state) # [32,4] batch_q = model(batch_state)
                
                # target_model for max q
                #with torch.no_grad():
                        #batch_q_target = batch_q.clone().detach()# clone vs copy # https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor
                if DOUBLE_FLAG:
                    target_q = predict_q.detach().clone()
                    batch_max_act = torch.argmax(model(batch_state_next))
                    target_q = batch_reward + \
                            (1-batch_done)*GAMMA*target_model(batch_state_next)[batch_index,batch_max_act]
                else:
                    target_q = predict_q.detach().clone()
                    target_q[range_index,batch_act] = batch_reward + \
                                (1-batch_done)*GAMMA*torch.amax(target_model(batch_state_next),dim=1)
                # pytorch sgd
                #loss = loss_func(batch_q,batch_q_target)   #seems smaller 
                if CLIP_FLAG:
                    target_q = (target_q - predict_q).clamp(-1,1)+predict_q
                    
                if PER_FLAG:
                    ReplayBuffer.priority_update((target_q-predict_q).cpu().detach().numpy(),batch_index)
                optimizer.zero_grad()
                loss = loss_func(predict_q,target_q)    
                loss = loss.clamp(-1,1)
                loss.backward()
                optimizer.step()                
                loss_list.append(float(loss))
                
                del batch_index,batch_state, batch_act, batch_reward, batch_state_next, batch_done 
            else:
                SKIP_STEP+=1
        # copy parameters==============================
        UPDATE_STEP += 1
        if  UPDATE_STEP >=  MODEL_UPDATE_STEP:
            UPDATE_STEP = 0
            #print("\nModel Bias:",model.advantage_fc2.bias.data)
            target_model.load_state_dict(model.state_dict()) 
        
        ob = ob_next
        step += 1
        frame_sum +=1
        
        max_q_list.append(max_q)
        reward_list.append(reward)
            
        if done:            
            break
    #==================================    
    ep += 1
    
    reward_sum.append(sum(reward_list))
    reward_ave.append(np.mean(reward_sum[-100:]))            
    max_q_max.append(max(max_q_list))
    
    log_file.write(f"\nEpoch: {ep} \tstep: {step}" 
                   f"\tframe_sum: {frame_sum}"
                   f"\tepsilon: {EPSILON:.3f}"
                   f"\tmax_q: {max_q_max[-1]:.3f}"
                   f"\tsum_rewards: {reward_sum[-1]:.3f}"
                   f"\tmean_rewards: {reward_ave[-1]:.3f}")
    if len(reward_ave) > 20:
        log_file.write(             
                   f"\tbest_mean_rewards: {max(reward_ave[20:]):.3f}")

    if len(loss_list):
        loss_sum.append(sum(loss_list))
        log_file.write(f"\tsum_loss: {loss_sum[-1]:.6f}")
        
    log_file.close()
    
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
print('\nShow the reward every ep')
plt.figure()
plt.plot(reward_ave,label = 'reward_ave')
plt.savefig(DIR + '/rewards.png')

plt.figure()
plt.plot(loss_sum)
plt.savefig(DIR + '/loss.png')

plt.figure()
plt.plot(max_q_max)
plt.savefig(DIR + '/max_q.png')

plt.figure()
plt.plot(reward_ave,label = 'reward_ave')
plt.plot(loss_sum,  label = 'loss')
plt.plot(max_q_max, label='max_q')
plt.legend(loc=2)
plt.savefig(DIR+ '/compare.png')

torch.save(model.state_dict(),os.path.join(DIR,'Model.pt'))
env.close()
print("Done")