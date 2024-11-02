# %%
import os
import imageio
import shutil
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from dm_wrapper import make_env

# %%
env = make_env("BreakoutNoFrameskip-v4")

# %%

ROOT_DIR = "Test_Wrapper"
DIR_PNG  = os.path.join(ROOT_DIR,"png")
try:
    os.mkdir(ROOT_DIR)
except:
    pass
try:
    os.mkdir(DIR_PNG)
except:
    pass

# %%

ob = env.reset()
images = []

for i in range(100):
    images.append(env.render(mode='rgb_array'))
    act = env.action_space.sample()
    ob_next, reward, done, info = env.step(act)
    
    plt.imsave(os.path.join(DIR_PNG,str(i)+".png"),np.concatenate(ob_next,axis=1))
    
    if done:
        break    
imageio.mimsave("Test_Wrapper/test.gif",images,fps=40)
# %%
images = []
for f in os.listdir(DIR_PNG):
    images.append(imageio.imread(os.path.join(DIR_PNG,f)))
imageio.mimsave(os.path.join(ROOT_DIR,'4in1.gif'),images,fps=50)
shutil.rmtree(DIR_PNG)
# %%
#print(ob_next._frames[0].shape,act,reward,done,info)
print(ob_next[0].shape)
print(ob_next._force().shape)
print(len(ob_next))

# %%
for i in range(len(ob_next)):
    plt.subplot(1,4,i+1)
    plt.imshow(ob_next[i],cmap='gray')
    plt.axis('off')
plt.show()  
# %%
print(np.concatenate(ob_next,axis=1).shape)
plt.imshow(np.concatenate(ob_next,axis=1))
# %%
class Replay:
    def __init__(self, MEMORY_SIZE = 5000, BATCH_SIZE = 32):
        self.BATCH_SIZE  = BATCH_SIZE
        self.MEMORY_SIZE = MEMORY_SIZE
        self.state_memo      = deque([],maxlen = MEMORY_SIZE)
        self.state_next_memo = deque([],maxlen = MEMORY_SIZE)
        self.act_memo    = deque([],maxlen = MEMORY_SIZE)
        self.reward_memo = deque([],maxlen = MEMORY_SIZE)
        self.done_memo   = deque([],maxlen = MEMORY_SIZE)
        
    def memo_append(self, ob, act,reward, ob_next, done):
        # a_set_memory = sars(a) : [ob, (act), reward, ob_next, done]
                                    
        self.state_memo.append(ob.concatenate())
        self.state_next_memo.append(ob_next.concatenate())        
        self.act_memo.append(act)
        self.reward_memo.append(reward)
        self.done_memo.append(done)
        
    def memo_len(self):
        return len(self.state_memo)
        
    def sample(self):
        batch_index = random.sample(range(self.memo_len()),self.BATCH_SIZE)
        
        batch_state      = np.array(self.state_memo)[batch_index]
        batch_state_next = np.array(self.state_next_memo)[batch_index]
        batch_act        = np.array(self.act_memo)[batch_index]
        batch_reward     = np.array(self.reward_memo)[batch_index]
        batch_done       = np.array(self.done_memo)[batch_index]
        
        return batch_state, batch_act, batch_reward, batch_state_next, batch_done
# %%
memo = Replay()

# %%
ob = env.reset()

for i in range(100):
    env.render()
    act = env.action_space.sample()
    ob_next, reward, done, info = env.step(act)
    
    memo.memo_append(ob,act,reward,ob_next,done)
    
    if done:
        break    
# %%
print(memo.memo_len())
# %%
batch_index = random.sample(range(memo.memo_len()),memo.BATCH_SIZE)
print(batch_index)
# %%
batch_state, batch_act, batch_reward, batch_state_next, batch_done = memo.sample()

# %%
print(batch_state.shape)
# %%
print(1-batch_done)
#%%
batch_q = np.random.randint(4,size=[32,4]).astype('float')
batch_q_next = np.random.randint(4,size=[32,4]).astype('float')
print(batch_q.shape)

#  %%
batch_q_new =np.copy(batch_q)
max_action = np.argmax(batch_q,axis=1)
batch_q_new[range(memo.BATCH_SIZE),batch_act] = batch_reward + (1-batch_done)*0.9*batch_q_next[range(memo.BATCH_SIZE),max_action]
#%%
n=3
# %%
print(batch_q_new[n])
# %%
print(batch_q[n],max_action[n])
print(batch_q_next[n],batch_q_next[n,max_action[n]])
# %%
print(batch_reward[n]+(1-batch_done[n])*0.9*batch_q_next[n,max_action[n]])
print(batch_q_new[n,batch_act[n]])
print(batch_q[n,batch_act[n]])
# %%


# %%
