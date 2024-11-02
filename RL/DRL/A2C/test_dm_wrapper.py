# %%
from dm_wrapper import make_env

env_name = "PongNoFrameskip-v4"

env = make_env(env_name)
# %%
ob = env.reset()
ACNet = AC_CNN()
# %%
ob,reward,done, info = env.step(env.action_space.sample())

import matplotlib.pyplot as plt
import numpy as np
plt.imshow(np.concatenate(ob._frames, axis = -1)[:,:,0])
# %%
plt.imshow(np.concatenate(ob._frames, axis = -1)[:,:,1])
# %%
plt.imshow(np.concatenate(ob._frames, axis = -1)[:,:,2])
# %%
plt.imshow(np.concatenate(ob._frames, axis = -1)[:,:,3])
# %%
plt.imshow(ob.concatenate()[1])
# %%
for i in range(20):
    
    ob,reward,done, info = env.step(env.action_space.sample())
    
    
    if done:
        break