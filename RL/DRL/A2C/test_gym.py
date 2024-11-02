# %%
import gym
env_name = "MountainCar-v0"
env_name = "LunarLander-v2"
env_name = "MountainCarContinuous-v0"
env_name = "LunarLanderContinuous-v2"
env = gym.make(env_name)

# %%
ob = env.reset()
print(ob)
# %%
env.render()
# %%
env.close()
# %%
