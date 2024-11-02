# %%
import torch
import numpy as np

from collections import namedtuple,deque

from model import ActorCritic
from dm_wrapper import make_env
# %%
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Running on GPU")
else:
    DEVICE = torch.device('cpu')
    print("Running on CPU")
    
# %%
env = make_env("PongNoFrameskip-v4")
N_ACT = env.action_space.n
N_OB  = env.observation_space.shape
# %%
ob = env.reset()
ac = ActorCritic(N_OB[2],N_ACT).to(DEVICE)


# %%
def get_action(ob):
    v,p = ac(ob_to_s(ob).to(DEVICE))
    p_np = p.cpu().detach().view(-1).numpy()
    return v.item(), p_np, np.random.choice(N_ACT,p=p_np)
def ob_to_s(ob):
    return torch.Tensor(ob.concatenate()).permute(2,0,1).view(-1,N_OB[2],N_OB[0],N_OB[1])

# %%
Experience = namedtuple('Transition',
                        ['reward','value','policy'])    

#%%time
T_MAX = 20


ob = env.reset()
t = 0
r_v_p  = []
t_start = t
while(1):
    #env.render()
    value, policy, act = get_action(ob)
    next_ob ,reward,done,info = env.step(act)
    
    if done: value, policy = 0, np.zeros(N_ACT)
    r_v_p.append(Experience(reward,value,policy[act]))
    
    t += 1
    ob = next_ob
    if (t - t_start)== T_MAX:
        break
    
    if done:
        break
        
env.close()
# %%
G = 0
GAMMA = 0.99
while len(r_v_p):
    reward, value, policy = r_v_p.pop()
    G =  value* GAMMA + reward
    