# %%
import os
import imageio
import datetime
import random

import matplotlib.pyplot as plt
import numpy as np

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dm_wrapper import make_env
from utils import Args, DDQN, PReplay, Logfile

# %%
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Running on GPU")
else:
    DEVICE = torch.device('cpu')
    print("Running on CPU")
# %%        
args = Args

FRAME_END = args.FRAME_END
FRAME_T   = args.FRAME_T

GAMMA   = args.GAMMA
EPSILON = args.EPSILON
EPSILON_END   = args.EPSILON_END

LEARNING_RATE = args.LEARNING_RATE

MEMORY_SIZE   = args.MEMORY_SIZE
MEMORY_SAMPLE_START = args.MEMORY_SAMPLE_START

MODEL_UPDATE_STEP   = args.MODEL_UPDATE_STEP
TRAIN_SKIP_STEP    = args.TRAIN_SKIP_STEP

BATCH_SIZE = args.BATCH_SIZE

RENDER     = args.RENDER
GIF_MAKE   = args.GIF_MAKE
ENV_NAME   = args.ENV_NAME
# %%
env = make_env(ENV_NAME)
N_ACT = env.action_space.n
N_OB  = env.observation_space.shape # 84.84.4

# %%
FILENAME = os.path.splitext(os.path.basename(__file__))[0]
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
    
# %%
log_file = Logfile(DIR+'/log.txt')
log_file.write(f"FRAME_END:{FRAME_END}, FRAME_T:{FRAME_T}\n"   
               f"EPSILON:{EPSILON}, EPSILON_END:{EPSILON_END}\n"   
               f"GAMMA: {GAMMA},  LEARNING_RATE:{LEARNING_RATE} \n"  
               f"MEMORY_SIZE:{MEMORY_SIZE} ,MEMORY_SAMPLE_START:{MEMORY_SAMPLE_START}\n" 
               f"MODEL_UPDATE_STEP:{MODEL_UPDATE_STEP}, TRAIN_SKIP_STEP:{TRAIN_SKIP_STEP}\n"  
               f"BATCH_SIZE:{BATCH_SIZE}\n")
log_file.close()
# %%
ReplayBuffer = PReplay(MEMORY_SIZE = MEMORY_SIZE,BATCH_SIZE = BATCH_SIZE)

model = DDQN(N_OB[2],N_ACT).to(DEVICE)
target_model = DDQN(N_OB[2],N_ACT).to(DEVICE)
target_model.load_state_dict(model.state_dict())
        
optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)
loss_func = nn.MSELoss()
  
    
def get_action(state): # get action with epsilon greedy
    # state modify for dm_wrapper
    with torch.no_grad():
        q = model(state)
    if np.random.rand() < EPSILON:            
        return np.random.randint(N_ACT), torch.amax(q,dim=1)
    return torch.argmax(q), torch.amax(q,dim=1)

    
def replay_sample_tensor():   
    batch_state, batch_act, batch_reward, batch_state_next, batch_done = ReplayBuffer.sample()

    batch_state      = torch.Tensor(batch_state).permute(0,3,1,2).to(DEVICE)
    batch_state_next = torch.Tensor(batch_state_next).permute(0,3,1,2).to(DEVICE)
    
    
    batch_act    = torch.Tensor(batch_act).type(torch.LongTensor).to(DEVICE)
    batch_reward = torch.Tensor(batch_reward).to(DEVICE)
    batch_done   = torch.Tensor(batch_done).to(DEVICE)
    return batch_state, batch_act, batch_reward, batch_state_next, batch_done
               
# %%
epsilon_decay = (EPSILON-EPSILON_END)/FRAME_T
loss_sum   = []
max_q_max  = []
reward_ave = []
reward_sum = []
frame_sum = 0
ep = 0

TEST = False
UPDATE_STEP = 0
SKIP_STEP = 0 
# %%
while(1):
    
    log_file.open()

    loss_list   = []
    reward_list = []
    max_q_list  = []
    step        = 0
    if GIF_MAKE: images = []
    ob = env.reset()
    
    
    #======================================================
    while(1):
        if TEST:
            EPSILON = 0
        else:
            EPSILON = max([EPSILON_END,EPSILON-epsilon_decay])
            
        if RENDER:
            env.render()
        if GIF_MAKE:
            images.append(env.render(mode='rgb_array'))

        state = torch.Tensor(ob.concatenate()).view(-1,N_OB[2],N_OB[0],N_OB[1]).to(DEVICE)
        act, max_q = get_action(state)
        ob_next, reward, done, info = env.step(act)

        #ob_next_array = ob_next.concatenate()        
        ReplayBuffer.memo_append(ob.concatenate(),act,reward,ob_next.concatenate(),done)
        
        # batch train==============================
        if ReplayBuffer.memo_len() > MEMORY_SAMPLE_START:
            if SKIP_STEP > TRAIN_SKIP_STEP:
                SKIP_STEP = 0
                
                batch_state, batch_act, batch_reward, batch_state_next, batch_done = replay_sample_tensor()
                
                range_index = torch.arange(0,BATCH_SIZE,device=DEVICE)
                batch_act = batch_act.type(torch.LongTensor).to(DEVICE)
                # model for q now
                predict_q = model(batch_state)[range_index,batch_act] # [32,4] batch_q = model(batch_state)
                
                # target_model for max q
                with torch.no_grad():
                    #batch_q_target = batch_q.clone().detach()# clone vs copy # https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor

                    batch_max_act = torch.argmax(model(batch_state_next),axis=1) # [32]
                    
                    target_q = batch_reward + \
                                (1-batch_done)*0.9*target_model(batch_state_next)[range_index,batch_max_act]
                # pytorch sgd
                #loss = loss_func(batch_q,batch_q_target)   #seems smaller 
                optimizer.zero_grad()
                loss = loss_func(predict_q,target_q)    
                #loss = loss.clamp(-1,1)
                loss.backward()
                optimizer.step()                
                loss_list.append(float(loss))
                
                del batch_state, batch_act, batch_reward, batch_state_next, batch_done 
            else:
                SKIP_STEP+=1
        # copy parameters==============================
        if UPDATE_STEP > MODEL_UPDATE_STEP:
            UPDATE_STEP = 0
            #print("\nModel Bias:",model.advantage_fc2.bias.data)
            target_model.load_state_dict(model.state_dict()) 
        else:
             UPDATE_STEP += 1  
        
        ob = ob_next
        step += 1
        frame_sum +=1
        
        max_q_list.append(float(max_q))
        reward_list.append(reward)
            
        if done:            
            break
    #==================================    
    ep += 1
    
    reward_sum.append(sum(reward_list))
    reward_ave.append(sum(reward_sum[-40:])/min(40,len(reward_sum)))            
    max_q_max.append(max(max_q_list))
    
    log_file.write(f"\nEpoch: {ep} \tstep: {step}" 
                   f"\tframe_sum: {frame_sum}"
                   f"\tepsilon: {EPSILON:.3f}"
                   f"\tmax_q: {max_q_max[-1]:.3f}"
                   f"\tmean_rewards: {reward_ave[-1]:.3f}"
                   f"\tsum_rewards: {reward_sum[-1]:.3f}")

    if len(loss_list):
        loss_sum.append(sum(loss_list))
        log_file.write(f"\tave_loss: {loss_sum[-1]:.6f}")
        
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
print('Show the reward every ep')
plt.figure()
plt.plot(reward_ave)
plt.savefig(DIR + '/rewards.png')

plt.figure()
plt.plot(loss_sum)
plt.savefig(DIR + '/loss.png')

plt.figure()
plt.plot(max_q_max)
plt.savefig(DIR + '/max_q.png')
   

torch.save(model.state_dict(),os.path.join(DIR,'Model.pt'))
print("Done")