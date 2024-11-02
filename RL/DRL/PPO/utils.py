import argparse
def get_args():
    parser = argparse.ArgumentParser(description='PPO Hyperparameters Adjust')
    parser.add_argument("--EPOCHS",default=2000,
                                    type=int,
                                    help="Whole program epochs")
    
    parser.add_argument("--UPDATE_EP",default=5,
                                    type=int,
                                    help="Trainning to update model every x epochs")
    
    parser.add_argument("--EVAL_FREQUENCY",default=1,
                                    type=int,
                                    help="Test the model every x epochs")
    
    parser.add_argument("--STEPS_PER_EP",default=2048,
                                    type=int,
                                    help="Collect constant steps per ep, controled by ONE_EP_FLAG")
    
    parser.add_argument("--WORKER_NUM",default=4,
                                    type=int,
                                    help="Parallel worker number to collect experience, total collections equal STEPS_PER_EP ")
    
    parser.add_argument("--ONE_EP_FLAG",default=False,
                                    action="store_true",
                                    help="Flag to collect steps per ep until env ends, Fasle to collect constant steps, controlled by STEPS_PER_EP ")
    
    parser.add_argument("-lr","--LR",default=3e-4,
                                    type=float,
                                    help="Learning rate of model under Adam")
    
    parser.add_argument("--EPSILON",default=0.2,
                                    type=float,
                                    help="Loss clip parameter: rs(1-epsilon,1+epsilon)*Advantage")
    
    parser.add_argument("--GAMMA",default=0.995,
                                    type=float,
                                    help="TD error parameter: r + GAMMA * V(s') - V(s)")
    
    parser.add_argument("--LAMBDA",default=0.97,
                                    type=float,
                                    help="GAE parameter: advantage = sum( LAMBDA * GAMMA * td_error)")
    
    parser.add_argument("--fc_n",default=[64,64],
                                    nargs="+",
                                    type=int,
                                    help="Hidden layer number of Network")
    
    parser.add_argument("--COEF_ENTROPY",default=0.1,
                                    type=float,
                                    help="Coefficience of entropy loss")
    
    parser.add_argument("--COEF_VAL",default=0.5,
                                    type=float,
                                    help="Coefficience of value loss")
    
    parser.add_argument("--BATCH_SIZE",default=64,
                                    type=int,
                                    help="Batchsize of experience sample when trainning")
    
    parser.add_argument("-r","--RENDER_FLAG",default=False,
                                    action="store_true",
                                    help="Flag to render when testing")
    
    parser.add_argument("--TENSORBOARD_FLAG",default=True,
                                    action="store_false",
                                    help="Flag to use tensorboard")
    
    parser.add_argument("-n","--NORMALIZE_FLAG",default=True,
                                    action="store_false",
                                    help="Flag to normalize observations for model")
    
    parser.add_argument("--DEVICE",default=None,
                                    choices=['cpu','cuda:0'],
                                    help="Specify using 'cpu' or 'gpu', otherwise use gpu is capable")
    
    parser.add_argument("--env_name",default="LunarLanderContinuous-v2",
                                    choices=["LunarLanderContinuous-v2",
                                             "MountainCarContinuous-v0",
                                             "BipedalWalkerHardcore-v3"],
                                    help="Choose name of environment")
    args = parser.parse_args()
    
    return args

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