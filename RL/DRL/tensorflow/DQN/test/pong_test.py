from os import access
import gym
import pdb

env = gym.make("Breakout-v0")
ob = env.reset()

act_repeat = 4
for i in range(300):
    env.render()
    #act = env.action_space.sample() 
    print(act_repeat)
    if act_repeat == 4:
        act = int(input("select Action"))
        act_repeat = 0
    else:
        #pdb.set_trace()
        act_repeat +=1

    print(i,act)

    ob, reward,done,info = env.step(act)

    # pdb.set_trace()
    print(reward,done,info)

    if done:
        break

