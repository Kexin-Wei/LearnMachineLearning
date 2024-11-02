
import numpy as np
import gym
import matplotlib.pyplot as plt
class env_fl():
    def __init__(self):
        pass

    def render_fig(self,name):
        s0=int(self.observation)/4
        s1=int(self.observation)%4
        fig=plt.figure(figsize=(4,4))
        plt.xlim([0,4])
        plt.ylim([0,4])
        ax=fig.gca()
        ax.set_xticks(np.arange(0,5,1))
        ax.set_yticks(np.arange(0,5,1))
        plt.grid()  #set grid

        ax = fig.add_subplot(111)
        for T in self.Terminal:
            plt.gca().add_patch(plt.Rectangle((T[0],T[1]),1,1,color="black"))
        

        for i in range(4):
            for j in range(4):
                if self.Policy[i,j,0]!=0: plt.annotate(text="",xy=(i,j+0.5),xytext=(i+0.5,j+0.5),arrowprops={"arrowstyle":"->"}) #left
                if self.Policy[i,j,1]!=0: plt.annotate(text="",xy=(i+0.5,j+1),xytext=(i+0.5,j+0.5),arrowprops={"arrowstyle":"->"}) #up
                if self.Policy[i,j,2]!=0: plt.annotate(text="",xy=(i+1,j+0.5),xytext=(i+0.5,j+0.5),arrowprops={"arrowstyle":"->"}) #right
                if self.Policy[i,j,3]!=0: plt.annotate(text="",xy=(i+0.5,j),xytext=(i+0.5,j+0.5),arrowprops={"arrowstyle":"->"}) #down                          
        plt.savefig(name)
        plt.close()
        

def test1(gym_name): # random step
    import gym
    env = gym.make(gym_name) # CartPole-v0, MountainCar-v0, 'FrozenLake-v0'
    env.reset()
    for _ in range(1000):
        env.render()
        _,_,done,_=env.step(env.action_space.sample()) # take a random action
        if done:
            return
    env.close()

def test2(gym_name): # show up model
    import gym
    env = gym.make(gym_name) #'Copy-v0'
    env.reset()
    env.render()

def test2_2(gym_name): #
    import gym
    env = gym.make(gym_name) # SpaceInvaders-v0, MsPacman-v0
    env.reset()
    env.render()

def test3(gym_name): # check what is in observation
    import gym
    env = gym.make(gym_name) # Breakout-v0, FrozenLake-v0
    end_steps=[]
    for i_episode in range(20):
        observation = env.reset()
        for t in range(10**6):
            env.render()
            print(t)
            action = env.action_space.sample() # take a random action
            #print(action)
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                end_steps.append(t+1)
                break
    plt.plot(end_steps)
    plt.show()
    env.close()

def test4(env_name):
    env = gym.make(env_name)
    highscore = 0
    point_record=[]
    for i_episode in range(20): # run 20 episodes
        observation = env.reset()
        points = 0 # keep track of the reward each episode
        while True: # run until episode is done
            env.render()
            action = 1 if observation[2] > 0 else 0 # if angle if positive, move right. if angle is negative, move left
            observation, reward, done, info = env.step(action)
            points += reward
            point_record.append(points)
            if done:
                break
    plt.plot(point_record)
    plt.show()
    env.close()           
        
    

if __name__=="__main__":
    test3('Breakout-v0')