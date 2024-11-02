# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# %%
class ACNET_FC(nn.Module):
    # discrete actor critic network
    def __init__(self,lr, IN_DIMS, N_ACT,  FC1_DIMS = 128, FC2_DIMS = 128):
        super().__init__()
        
        self.N_ACT = N_ACT
        
        self.fc1 = nn.Linear(IN_DIMS,FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS,FC2_DIMS)

        self.policy = nn.Linear(FC2_DIMS,N_ACT)
        self.value  = nn.Linear(FC2_DIMS,1)                                
        
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.to(self.device)
        
    def forward(self, ob):
        state = torch.Tensor(ob).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        p = F.softmax(self.policy(x),dim=0)
        v = self.value(x)
        
        return v,p
# %%
class ACNET_FC_FLAT(nn.Module):
    def __init__(self,lr,N_ACT,*args):
        super().__init__()
        if args:
            fc_num = args[0]
            assert len(fc_num) == 2
            fc1, fc2 = fc_num[0],fc_num[1]
        else:
            fc1, fc2 = 256,256
            
        self.fc1 = nn.Linear(84*84,fc1)
        self.fc2 = nn.Linear(fc1,fc2)

        self.policy = nn.Linear(fc2,N_ACT)
        self.value  = nn.Linear(fc2,1)                                
        
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self,ob):
        state = torch.Tensor(ob).reshape(-1,84*84).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        p = F.softmax(self.policy(x),dim=0)
        v = self.value(x)
        
        return v,p
        
# %%
class ACNET_CNN(nn.Module):
    # structure 2 head network https://www.datahubbs.com/two-headed-a2c-network-in-pytorch/
    def __init__(self,lr,IN_CHANNELS,N_ACT):
        super().__init__()
        
        self.conv1 = nn.Conv2d(IN_CHANNELS,16,kernel_size=8,stride=4)
        self.conv2 = nn.Conv2d(16,32,kernel_size=4,stride=2)
        self.fc    = nn.Linear(9*9*32,256)
        # why share https://discuss.pytorch.org/t/actor-critic-implementation-problem/12782
        self.value  = nn.Linear(256,1)
        self.policy = nn.Linear(256,N_ACT)
        
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.to(self.device)
    
    def forward(self,ob):
        state = torch.Tensor(ob.concatenate()).view(-1,4,84,84).to(self.device)
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.shape[0],-1)
        x = F.relu(self.fc(x))
        
        v = self.value(x)
        p = F.softmax(self.policy(x),dim=0)
        return v,p
    
# %%    
class Actor(nn.Module):
    # discrete actor critic network
    def __init__(self,lr, IN_DIMS, N_ACT,  FC1_DIMS = 128, FC2_DIMS = 128):
        super().__init__()
        
        self.N_ACT = N_ACT
        
        self.fc1 = nn.Linear(IN_DIMS,FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS,FC2_DIMS)
        self.policy = nn.Linear(FC2_DIMS,N_ACT)                                   
        
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.to(self.device)
        
    def forward(self, ob):
        state = torch.Tensor(ob).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))        
        p = F.softmax(self.policy(x),dim=0)                
        return p  
    
class Critic(nn.Module):
    # discrete actor critic network
    def __init__(self,lr, IN_DIMS,  FC1_DIMS = 128, FC2_DIMS = 128):
        super().__init__()
        
        
        self.fc1 = nn.Linear(IN_DIMS,FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS,FC2_DIMS)
        self.value  = nn.Linear(FC2_DIMS,1)                                     
        
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.to(self.device)
        
    def forward(self, ob):
        state = torch.Tensor(ob).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))        
        v = self.value(x)        
        return v
 
    
    
    
# from dm_wrapper import make_env
# env = make_env("PongNoFrameskip-v4")
# N_ACT = env.action_space.n
# N_OB  = env.observation_space.shape
# # %%
# ob = env.reset()
# ac = ACNET_CNN(N_OB[2],N_ACT)

# def ob_to_s(ob):
#     return torch.Tensor(ob.concatenate()).view(-1,N_OB[2],N_OB[0],N_OB[1])
# #%%
# for i in range(100):
#     env.render()
#     act = env.action_space.sample()
#     ob_next,reward,done,info = env.step(act)
#     v, p = ac(ob_to_s(ob))
#     ob = ob_next
#     if done :
#         break
# %%
