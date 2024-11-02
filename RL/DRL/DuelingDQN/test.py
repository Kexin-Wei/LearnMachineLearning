# %%
import gym
from imageio.core.util import asarray

from utils import Replay
from dm_wrapper import make_env

BATCH_SIZE  = 32
MEMORY_SIZE = 10_000
ENV_NAME = "PongNoFrameskip-v4"
# %%
env = make_env(ENV_NAME)
N_ACT = env.action_space.n
N_OB  = env.observation_space.shape # 84.84.4

ReplayBuffer = Replay(MEMORY_SIZE = MEMORY_SIZE,BATCH_SIZE = BATCH_SIZE)

# %%
while(1):
    ob = env.reset()
    while(1):
        
        act = env.action_space.sample()
        
        ob_next, reward, done, info = env.step(act)
        
        ReplayBuffer.memo_append(ob.concatenate(),act,reward,ob_next.concatenate(),done)
        
        if ReplayBuffer.memo_len()>1000:
            batch_state, batch_act, batch_reward, batch_state_next, batch_done = ReplayBuffer.sample()
            
        if done:
            break
# %%
def args_test(*args):
    if args:
        print(args)
        assert len(args[0])==2
    else:
        print("No args")
# %%