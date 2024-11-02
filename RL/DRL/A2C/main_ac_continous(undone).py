# %%
# one step advantage actor critic
import gym
import os
import platform

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
# %%
lr = 1e-6
COEF_ENT   = 0.01
COEF_VALUE = 0.5
GAMMA = 0.99
fc_n = [1024,1024]
RENDER_FLAG      = False
TENSORBOARD_FLAG = True
EPOCHS = 2000

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# %%
env_name = "LunarLanderContinuous-v2"
env = gym.make(env_name)
N_OB  = env.observation_space.shape[0]
N_ACT = env.action_space.shape[0]

# %%
FILENAME = os.path.splitext(os.path.basename(__file__))[0]
OS = "mac" if platform.system() == "Darwin" else "linux"
DIR = f"test_{OS}_{FILENAME}_Entropy_{env_name}"
try:
    os.makedirs(DIR)
except:
    print(f"Failed to open folder {DIR}")

# %%
class ActorCritic(nn.Module):
    def __init__(self,INPUT_DIM,OUTPUT_DIM,fc_n = [64,64],device = 'cpu'):
        super().__init__()
        
        self.n = len(fc_n)
        self.fc_n = fc_n
        self.INPUT_DIM = INPUT_DIM
        self.OUTPUT_DIM = OUTPUT_DIM
        
        self.actor = self.create_fc_net(self.OUTPUT_DIM)
        self.logstd   = nn.Parameter(torch.ones(self.OUTPUT_DIM)) 
        
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
        
        net = nn.Sequential(*layers)
        def initial(m):
            if type(m) == nn.Linear:
                #https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
                #nn.init.xavier_normal_(m.weight)
                nn.init.normal_(m.weight,mean=0,std=0.1)
                nn.init.constant_(m.bias,0.1)
        net.apply(initial)
        return net
    
    def get_act(self,ob):
        state = torch.Tensor(ob).view(-1,self.INPUT_DIM).to(self.device)
        dist  = self.dist_(state)
        act   = dist.sample()        
        return act[0].cpu().detach().numpy()
# %%
ACNet = ActorCritic(N_OB,N_ACT,fc_n=fc_n,device=device) 

comment = f"lr_{lr}"
for i in range(len(fc_n)):        
        comment += f"_fc{i+1}_{fc_n[i]}"

if TENSORBOARD_FLAG:
    writer = SummaryWriter(os.path.join(DIR,comment))
    
TEST = False
ep=0
reward_list = []
loss_list   = []
def get_action(ob):   
    v, p = ACNet(ob)
    dist = torch.distributions.Categorical(p) # discrete distribution with N_ACTs probabilities
    act  = dist.sample()
    return act.item()

def train(ob, act, reward, next_ob, done):
    v,p = ACNet(ob)
    next_v , next_p = ACNet(next_ob)

    act_tensor = torch.Tensor([act]).to(ACNet.device)
    # TD error
    td_error  = torch.Tensor([reward]).to(ACNet.device)\
                + (1-done)*GAMMA* next_v - v
                
    # actor loss <- negative *log(p(ob,act)*TD error
    dist = torch.distributions.Categorical(p) # discrete distribution with N_ACTs probabilities
    actor_loss   = - dist.log_prob(act_tensor) * td_error
    entropy_loss = - COEF_ENT*dist.entropy() # already has - 
    critic_loss  = COEF_VALUE*td_error**2
    total_loss = actor_loss+entropy_loss+critic_loss
    
    ACNet.optimizer.zero_grad()
    total_loss.backward()
    ACNet.optimizer.step()
    
    return total_loss.item()

while(1):
    ob = env.reset()
    
    reward_sum = 0
    
    if TEST: images = []
    #===============================
    while(1):
        
        if RENDER_FLAG:
            env.render()
        if TEST:
            images.append(env.render(mode='rgb_array'))
            
        act = get_action(ob)
        
        next_ob, reward, done, info = env.step(act)
        
        #reward = 1 if reward>0 else 0
        loss = train(ob, act, reward, next_ob, done)
        
        ob = next_ob
        reward_sum += reward
        
        if done:
            break
    #========================            
    if TEST:
        print(f"Final Test: reward:{reward_sum}")
        break
    
    ep += 1
    reward_list.append(reward_sum)
    loss_list.append(loss)
    
    print(f"Epoch:{ep} \t reward: {reward_sum:.2f} \t best reward:{max(reward_list):.2f} \t loss {loss_list[-1]:.3f}")
    
    if TENSORBOARD_FLAG:
        writer.add_scalar("Reward",reward_sum,ep)
        writer.flush()
    
    if ep>EPOCHS:
        TEST = True
    

if TEST:
    imageio.mimsave(f"{DIR}/{comment}_{reward_sum}.gif",images,fps=50)
    
if TENSORBOARD_FLAG:
    writer.close()

plt.figure()
plt.plot(reward_list)
plt.title("Reward")
plt.savefig(f"{DIR}/{comment}_reward_max{max(reward_list):.2f}.png")
env.close()

# %%