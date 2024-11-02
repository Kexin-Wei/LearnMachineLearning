import numpy as np

class Replay:
    def __init__(self, BATCH_SIZE = 32):
        
        self.BATCH_SIZE = BATCH_SIZE
        
        self.obs  = []
        self.obs_ = []
        self.acts = []
        self.dones   = []
        self.rewards = []
        self.advantages = []
        
    @property
    def len(self):
        return len(self.obs)
    
    def append(self,obs,obs_,acts,dones,rewards,advantages):
        self.obs.extend(obs)
        self.obs_.extend(obs_)
        self.acts.extend(acts)
        self.dones.extend(dones)
        self.rewards.extend(rewards)
        self.advantages.extend(advantages)
    
    def sample(self):
        indexs = np.random.choice(range(self.len),
                    self.BATCH_SIZE,replace=False)
        obs  = []
        obs_ = []
        acts = []
        dones   = []
        rewards = []
        advantages = []
        for idx in indexs:
            obs.append(self.obs[idx])
            obs_.append(self.obs_[idx])
            acts.append(self.acts[idx])
            dones.append(self.dones[idx])
            rewards.append(self.rewards[idx])
            advantages.append(self.advantages[idx])
        
        return obs,obs_,acts,dones,rewards,advantages
    
    def return_all(self):
        return self.obs, self.obs_, self.acts,\
               self.dones,self.rewards,self.advantages
               
               
# %%
from collections import deque
class ReplayDeque:
    def __init__(self,MEMORY_SIZE=int(1e6), BATCH_SIZE = 64):
            
            self.BATCH_SIZE = BATCH_SIZE
            
            self.obs  = deque([],maxlen=MEMORY_SIZE)
            self.obs_ = deque([],maxlen=MEMORY_SIZE)
            self.acts = deque([],maxlen=MEMORY_SIZE)
            self.dones   = deque([],maxlen=MEMORY_SIZE)
            self.rewards = deque([],maxlen=MEMORY_SIZE)
        
    @property
    def len(self):
        return len(self.obs)
    
    def add(self,ob,ob_,act,done,reward):
        # only data for timestep
        self.obs.append(ob)
        self.obs_.append(ob_)
        self.acts.append(act)
        self.dones.append(done)
        self.rewards.append(reward)
        
    def append(self,obs,obs_,acts,dones,rewards):
        # must requires a list for input
        self.obs.extend(obs)
        self.obs_.extend(obs_)
        self.acts.extend(acts)
        self.dones.extend(dones)
        self.rewards.extend(rewards)
    
    def sample(self):
        indexs = np.random.choice(range(self.len),
                    self.BATCH_SIZE,replace=False)
        obs  = []
        obs_ = []
        acts = []
        dones   = []
        rewards = []
        for idx in indexs:
            obs.append(self.obs[idx])
            obs_.append(self.obs_[idx])
            acts.append(self.acts[idx])
            dones.append(self.dones[idx])
            rewards.append(self.rewards[idx])
        
        return obs,obs_,acts,dones,rewards
    
    def return_all(self):
        return list(self.obs),list(self.obs_), list(self.acts),\
               list(self.dones),list(self.rewards)
# %%               
from collections import deque,namedtuple

memory = namedtuple('Experience', ['state','next_state', 'action','done', 'reward'])
class ReplayOneDeque:
    def __init__(self,MEMORY_SIZE = 5000, BATCH_SIZE = 64):
        self.BATCH_SIZE  = BATCH_SIZE
        self.MEMORY_SIZE = MEMORY_SIZE
        
        self.memo = deque([],maxlen = MEMORY_SIZE)
        
    def append(self, ob, ob_, act, done,reward):
        # a_set_memory = sars(a) : [ob, (act), reward, ob_next, done]
        # ob modify for dm_wrapper                                    
        self.memo.append(memory(ob,ob_,act,done,reward))
    
    @property
    def len(self):
        return len(self.memo)
            
    def sample(self):
        batch_index = np.random.choice(range(len(self.memo)),self.BATCH_SIZE,replace=False)
        
        batch_ob,batch_ob_,batch_act,batch_done,batch_reward = zip(*[self.memo[idx] for idx in batch_index])
        return np.array(batch_index), \
                np.array(batch_ob),   \
                np.array(batch_ob_),  \
                np.array(batch_act),  \
                np.array(batch_done),  \
                np.array(batch_reward)
            
if __name__ == "__main__":
    buffer = ReplayOneDeque(BATCH_SIZE=2)
    buffer.append(np.random.rand(3),np.random.rand(3),np.random.rand(2),
                  np.random.rand(1),np.random.rand(1))
    buffer.append(np.random.rand(3),np.random.rand(3),np.random.rand(2),
                  np.random.rand(1),np.random.rand(1))
    buffer.append(np.random.rand(3),np.random.rand(3),np.random.rand(2),
                  np.random.rand(1),np.random.rand(1))
    print("Done")