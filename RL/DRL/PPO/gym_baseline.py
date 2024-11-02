import gym
import matplotlib.pyplot as plt
# average -220

env_name = "MountainCarContinuous-v0"
env_name = "LunarLanderContinuous-v2"

env_name = "BipedalWalkerHardcore-v3"
env = gym.make(env_name)
reward_list = []

for ep in range(300):
    ob = env.reset()
    done = 0
    reward_sum = 0
    step = 0
    while not done:
        env.render()
        ob,reward, done, infor= env.step(env.action_space.sample())
        #print(ob.max(),ob.min())
        reward_sum += reward
        step +=1
        
    reward_list.append(reward_sum)
    print(f"Epoch: {ep}"
          f"\tstep: {step}"
          f"\treward: {reward_sum:.2f}"
          f"\tmean reward: {sum(reward_list[-100:])/min(len(reward_list),100):.2f}"
          f"\tbest reward: {max(reward_list):.2f}")    

env.close()
plt.figure()
plt.plot(reward_list)
plt.title("Reward")
plt.show()
#plt.savefig(f"{DIR}/reward_list_and_max{max(reward_list):.2f}.png")
    