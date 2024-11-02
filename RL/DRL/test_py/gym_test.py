import gym
import matplotlib.pyplot as plt
import os
import pdb
import numpy as np

DIR = "gym_graph"

env = gym.make('Breakout-v0')
env.reset()
img = plt.imshow(env.render(mode='rgb_array')) # only call this once
print(env.render(mode='rgb_array').shape)

for i in range(10000):
    plt.imsave(os.path.join(DIR,str(2)+"_"+str(i)+'.png'),env.render(mode='rgb_array'))
    
    ob,reward,done,info = env.step(env.action_space.sample())
    pdb.set_trace()
    if done:
        break
env.close()
print("Done with ",i)