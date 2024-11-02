# %%
# one step advantage actor critic
import gym
import datetime
import os
import platform

from agent import A2C_GAE

import matplotlib.pyplot as plt
from itertools import product
from torch.utils.tensorboard import SummaryWriter

parameters = dict(
    lr  = [1e-7,1e-8,1e-6],
)
para_pairs = [v for v in parameters.values()]

RENDER_FLAG      = False
TENSORBOARD_FLAG = True
EPOCHS = 3000
GAMMA  = 0.99
LAMBDA = 0.5
COEF_VALUE = 0.5
COEF_ENT   = 0.01
# %%
env_name = "LunarLander-v2"
env = gym.make(env_name)

N_OB  = env.observation_space.shape[0]
N_ACT = env.action_space.n

FILENAME = os.path.splitext(os.path.basename(__file__))[0]
OS = "mac" if platform.system() == "Darwin" else "linux"
# %%
DIR = os.path.join(f"test_{OS}_{FILENAME}_{env_name}",
                   datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
try:
    os.makedirs(DIR)
except:
    print(f"Failed to open folder {DIR}")

# %%
for lr,fc1,fc2  in product(*para_pairs):
    if fc1>=fc2:
        A2C_GAE(lr,
        GAMMA,LAMBDA,
        COEF_VALUE,COEF_ENT,
        EPOCHS,DIR,
        env,N_OB, N_ACT,
        fc1,fc2,
        TENSORBOARD_FLAG=TENSORBOARD_FLAG,RENDER_FLAG=RENDER_FLAG)

# %%