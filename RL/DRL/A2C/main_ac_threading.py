# %%
# one step advantage actor critic
import gym
import datetime
import os
import platform
import time
import threading

from agent import A2C
from dm_wrapper import make_env
# %%
lr = 1e-6
COEF_ENT   = 0.01
COEF_VALUE = 0.5
GAMMA = 0.99
fc1=2048
fc2=2048
RENDER_FLAG      = False
TENSORBOARD_FLAG = True
EPOCHS = 200
# %%
env_name = "PongNoFrameskip-v4"
env = make_env(env_name)
N_OB  = env.observation_space.shape
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


"""
only in main thread works
tensorboard failed
threads = []
for fc1,fc2 in product(*para_pairs):
    if fc1>fc2:
        t = threading.Thread(target=A2C,args=[
            1e-6,0.99,
            env,N_OB,N_ACT,
            2048,1024,
            EPOCHS,DIR,
        ],kwargs={
            "TENSORBOARD_FLAG":TENSORBOARD_FLAG,
            "RENDER_FLAG":RENDER_FLAG
        })
        t.start()
        threads.append(t)

for t in threads:
    t.join()

finish = time.perf_counter()
print(f"Done with {round(finish-start,2)} s")"""