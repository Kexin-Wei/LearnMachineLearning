import numpy as np
import gym



def test(gym_name): #random step
    env=gym.make(gym_name)
    env.reset()
    for _ in range(1000):
        env.render()
        observation, reward, done, info=env.step(env.action_space.sample()) # take a random action
        if done:
            return
    env.close()

def test3(gym_name): # manual chose step
    env=gym.make(gym_name)
    env.reset()
    for _ in range(1000):
        env.render()
        action = int(input('Choose from 0-3'))
        observation, reward, done, info = env.step(action)
        if done:
            return
    env.close()

def test2():
    n=4
    epsilon=0.2
    p=np.random.default_rng().dirichlet(np.ones(n))*(1-epsilon)
    p+=epsilon/n
    print(p)

if __name__ == "__main__":
    test("Breakout-v0")


