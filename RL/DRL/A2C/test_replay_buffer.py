# %%
from buffer import PER_Replay

from dm_wrapper import make_env

env_name = "PongNoFrameskip-v4"

env = make_env(env_name)

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
N_ACT = env.action_space.n 
N_OB  = env.observation_space.shape
MEMORY_SIZE = 10000
BATCH_SIZE  = 32
# %%
replay_buffer = PER_Replay(MEMORY_SIZE= MEMORY_SIZE, BATCH_SIZE = BATCH_SIZE)

# %%
ob = env.reset()

for i in range(40):
    env.render()
    act = env.action_space.sample()
    
    ob_next, reward, done, info = env.step(act)
    replay_buffer.append(ob.concatenate(), act, reward, ob_next.concatenate(),done)
    
    if done:
        break
env.close()
# %%
%%time
index, states, acts, rewards, next_states, dones = replay_buffer.sample()

# %%
