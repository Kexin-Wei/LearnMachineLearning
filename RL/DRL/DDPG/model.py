import torch
import torch.nn as nn
import numpy as np

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

class PPO_AC(nn.Module):
    def __init__(self,INPUT_DIM,OUTPUT_DIM,fc_n = [64,64],
                        device = 'cpu',CONTINUOUS = True,
                        NORMALIZE = True):
        super().__init__()
        
        self.n = len(fc_n)
        self.fc_n = fc_n
        self.CONTINUOUS = CONTINUOUS
        self.NORMALIZE  = NORMALIZE
        self.INPUT_DIM  = INPUT_DIM
        self.OUTPUT_DIM = OUTPUT_DIM
        
        self.actor  = self.create_fc_net(self.OUTPUT_DIM)
        self.logstd = nn.Parameter(torch.ones(self.OUTPUT_DIM)) 
        
        self.critic = self.create_fc_net(1)
        
        self.device = device
        self.to(self.device)
        
    def create_fc_net(self,out):
        layers=[]
        layer_n = self.fc_n[:]
        layer_n.insert(0,self.INPUT_DIM)
        for i in range(self.n):
            layers.append(nn.Linear(layer_n[i],layer_n[i+1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(self.fc_n[-1],out))
        
        # net weight and bias initial
        net = nn.Sequential(*layers)
        def initial(m):
            if type(m) == nn.Linear:
                #https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
                #nn.init.xavier_normal_(m.weight)
                nn.init.normal_(m.weight,mean=0,std=0.1)
                nn.init.constant_(m.bias,0.1)
        net.apply(initial)
        return net
    
    
    
    def dist_(self,states):
        #https://bochang.me/blog/posts/pytorch-distributions/
        act_means = self.actor(states)
        if self.CONTINUOUS:
            if self.OUTPUT_DIM == 1:
                dist = Normal(act_means,torch.exp(self.logstd))
            else:
                dist = MultivariateNormal(act_means,torch.diag(torch.exp(self.logstd)))# covariance as diag matrix
            # has batchsize: 1 for get_act, batch for get_policy
        else:
            act_probs = torch.nn.functional.softmax(act_means)
            dist = Categorical(act_probs)
        
        # print(f"\t Mean: {act_means.detach().numpy()} \t std: {torch.exp(self.logstd).detach().numpy()}",end="")
        return dist
    
    def obs_to_states(self,obs):
        states = torch.Tensor(obs).view(-1,self.INPUT_DIM).to(self.device)
        if not self.NORMALIZE:
            return states
        states = (states - states.amin(1,keepdim=True))/(states.amax(1,keepdim=True)-states.amin(1,keepdim=True))
        return states
    
    def get_act(self,ob):        
        state = self.obs_to_states(ob)
        dist  = self.dist_(state)
        act   = dist.sample()        
        return act[0].cpu().detach().numpy()
    
    def get_policy(self,obs,acts):
        states = self.obs_to_states(obs)
        acts   = torch.Tensor(acts).to(self.device)
        
        dist = self.dist_(states)
        
        log_prob  = dist.log_prob(acts)
        entropies = dist.entropy()
        return dist,log_prob,entropies # tensor type
    
    def td_cal(self,obs,obs_,dones,rewards,GAMMA):
        states = self.obs_to_states(obs)
        states_ = self.obs_to_states(obs_)
        
        dones = torch.Tensor(dones).to(self.device) 
        rewards = torch.Tensor(rewards).to(self.device) 
        
        values  = self.critic(states).squeeze()
        values_ = self.critic(states_).squeeze()
        td_errors = rewards + (1-dones)*GAMMA * values_ - values
        
        return td_errors
    
    def adv_cal(self,obs,obs_,dones,rewards,GAMMA,LAMBDA):
        
        td_errors = self.td_cal(obs,obs_,dones,rewards,GAMMA).tolist() 
            
        advantages,adv= [],0
        for i in reversed(range(len(td_errors))):
            adv = td_errors[i] + GAMMA*LAMBDA*adv
            advantages.append(adv)

        return advantages
    
# %%    
class DDPG_AC(nn.Module):
    def __init__(self,INPUT_DIM,OUTPUT_DIM,fc_n = [400,300],device = 'cuda:0'):
        super().__init__()
        
        self.n = len(fc_n)
        self.fc_n = fc_n
        self.INPUT_DIM  = INPUT_DIM
        self.OUTPUT_DIM = OUTPUT_DIM
        
        self.actor  = self.create_actor()
        self.critic = self.create_critic()
        
        self.device = device
    
    def create_actor(self):
        # q network
        layers=[]
        layer_n = self.fc_n[:]
        layer_n.insert(0,self.INPUT_DIM)
        
        for i in range(self.n):
            layers.append(nn.Linear(layer_n[i],layer_n[i+1]))
            uniform_range = 1/np.sqrt(layer_n[i])
            nn.init.uniform_(layers[-1].weight,-uniform_range,uniform_range)
            nn.init.uniform_(layers[-1].bias,-uniform_range,uniform_range)
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(self.fc_n[-1],self.OUTPUT_DIM))
        nn.init.uniform_(layers[-1].weight,a=-3e-3,b=3e-3)
        nn.init.uniform_(layers[-1].bias,  a=-3e-3,b=3e-3)
        layers.append(nn.Tanh())
        
        # net weight and bias initial
        net = nn.Sequential(*layers)
        return net
    
    def create_critic(self):
        # policy network
        layers=[]
        layer_n = self.fc_n[:]
        layer_n.insert(0,self.INPUT_DIM+self.OUTPUT_DIM)
        
        for i in range(self.n):
            layers.append(nn.Linear(layer_n[i],layer_n[i+1]))
            uniform_range = np.sqrt(layer_n[i])
            nn.init.uniform_(layers[-1].weight,-uniform_range,uniform_range)
            nn.init.uniform_(layers[-1].bias,-uniform_range,uniform_range)
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(self.fc_n[-1],1))
        
        nn.init.uniform_(layers[-1].weight,a=-3e-3,b=3e-3)
        nn.init.uniform_(layers[-1].bias,  a=-3e-3,b=3e-3)
        # net weight and bias initial
        net = nn.Sequential(*layers)
        return net
    
    def get_act(self,ob):
        state = torch.Tensor(ob).to(self.device)
        act = self.actor(state)
        return act
    
    def get_act_numpy(self,ob):
        act = self.get_act(ob)
        return act.cpu().detach().numpy()
    
    def get_qs(self,obs,acts):
        states = torch.Tensor(obs).to(self.device)
        qs = self.critic(torch.cat((states,acts),dim=-1))
        return qs

# %%
import torch.nn.functional as F
class Actor(nn.Module):
    def __init__(self,INPUT_DIM,OUTPUT_DIM,fc_n=[400,300],device="cpu"):
        super().__init__()
        assert len(fc_n)==2, "Should be only 2 hidden layers"
        self.fc1 = nn.Linear(INPUT_DIM,fc_n[0])        
        self.fc2 = nn.Linear(fc_n[0],fc_n[1])
        self.out = nn.Linear(fc_n[1],OUTPUT_DIM)
        
        f1 = 1/np.sqrt(INPUT_DIM)
        nn.init.uniform_(self.fc1.weight,-f1,f1)        
        nn.init.uniform_(self.fc1.bias,-f1,f1)
        f2 = 1/np.sqrt(fc_n[0])
        nn.init.uniform_(self.fc2.weight,-f2,f2)
        nn.init.uniform_(self.fc2.bias,-f2,f2)
        
        nn.init.uniform_(self.out.weight,-3e-3,3e-3)
        nn.init.uniform_(self.out.bias,-3e-3,3e-3)
        
        self.device = device
        self.to(self.device)
        
    def forward(self,obs):
        obs = torch.Tensor(obs).to(self.device)
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.out(x))
        return x
    
class Critic(nn.Module):
    def __init__(self,INPUT_DIM,OUTPUT_DIM,fc_n=[400,300],device="cpu"):
        super().__init__()
        assert len(fc_n)==2, "Should be only 2 hidden layers"
        self.fc1 = nn.Linear(INPUT_DIM+OUTPUT_DIM,fc_n[0])
        self.fc2 = nn.Linear(fc_n[0],fc_n[1])
        self.out = nn.Linear(fc_n[1],1)
        
        f1 = 1/np.sqrt(INPUT_DIM+OUTPUT_DIM)
        nn.init.uniform_(self.fc1.weight,-f1,f1)        
        nn.init.uniform_(self.fc1.bias,-f1,f1)
        f2 = 1/np.sqrt(fc_n[0])
        nn.init.uniform_(self.fc2.weight,-f2,f2)
        nn.init.uniform_(self.fc2.bias,-f2,f2)
        
        nn.init.uniform_(self.out.weight,-3e-3,3e-3)
        nn.init.uniform_(self.out.bias,-3e-3,3e-3)
        
        self.device = device
        self.to(self.device)
        
    def forward(self,obs):
        obs = torch.Tensor(obs).to(self.device)
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
       
    
if __name__ == "__main__":
    ac = DDPG_AC(9,2)

    print("end")
# %%
