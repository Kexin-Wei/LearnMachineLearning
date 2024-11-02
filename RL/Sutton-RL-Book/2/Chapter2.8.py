# Created by Vac. 2020.7.21
# for Reinforcement Learning Book Chapter 2.8
# 10 armed Bandit problem
# gradient bandit programm: with / without Base line R_ave
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle

class Bandit:
    def __init__(self, BanditNum=10, q_mu=0., q_sigma=1.0, stationary=True, \
                    
                    Q0=5, TimeStep=1000, \
                    set=2000, StepSize=0.1,\
                    Baseline=True):
        self.n = BanditNum                          # number of the Bandit
        self.Q = [ Q0 for i in range(self.n)]     # action value                
        self.N = np.zeros(BanditNum)                # record the times of choosing a

        self.set=set            # run time set (for average)
        self.stat = stationary  # if the bandit value change at every try
        #self.epsilon = epsilon  # epsilon-greedy method for exploration
        self.q_mu = q_mu        # true value q*(a) ~ N(q_mu,q_sigma)
        self.q_sigma = q_sigma  

        self.Reward = np.zeros(TimeStep)    # 1000 record the reward for every try
        self.RewardSum = np.zeros(TimeStep) # sum the reward for each set
        self.StepSize = StepSize      # alpha parameter of target error a(R-Q)
        self.TimeStep = TimeStep    # running times in one set
        self.OptAction= np.zeros(TimeStep)

        self.H = np.zeros(self.n)           # the action preference H for Gradient Bandit Program
        self.p_a = np.ones(self.n)/self.n   # the policy function in soft max distribution
        self.RewardAve = 0              # the Baseline of the Gradient Bandit Programm
        self.RestA=range(self.n)        # prepare for action update
        self.baselineF=Baseline         # use baseline or not

    def reset(self):
        self.q_n = np.random.normal( self.q_mu, self.q_sigma, self.n)
        self.Q=np.zeros(self.n)
        self.N=np.zeros(self.n)
        self.H=np.zeros(self.n)
        self.p_a = np.ones(self.n)/self.n
    
    def RL(self): 
        if self.baselineF:
            print("Using baseline")
        else:
            print("Without baseline")
        for a in range(self.set):
        # start the time step
            self.reset()
            for t in range( self.TimeStep ):
                
                #True for stationary 
                if self.stat==False: 
                    self.q_n = np.random.normal( self.q_mu, self.q_sigma, self.n)
            
                
                self.A = np.random.choice(np.arange(self.n),p=self.p_a)   # choose the action by its probability
                #,p=self.p_a
                self.Reward[t] = self.q_n[self.A]                   # get the reward

                self.RewardAve = self.RewardAve+(self.Reward[t]-self.RewardAve)/(t+1)   # cal the average reward 

                if self.A == np.argmax(self.q_n):
                    self.OptAction[t]+=1
                
                #update action preferences
                self.RestA=np.delete(self.RestA,self.A) #get the a!=A
                
                if self.baselineF: # use baseline for probability cal
                    self.H[self.A]=self.H[self.A]+self.StepSize*(self.Reward[t]-self.RewardAve)*(1-self.p_a[self.A])
                    self.H[self.RestA]=self.H[self.RestA]-self.StepSize*(self.Reward[t]-self.RewardAve)*self.p_a[self.RestA]
                else:
                    self.H[self.A]=self.H[self.A]+self.StepSize*(self.Reward[t])*(1-self.p_a[self.A])
                    self.H[self.RestA]=self.H[self.RestA]-self.StepSize*(self.Reward[t])*self.p_a[self.RestA]
                
                self.Q[self.A] +=self.StepSize*(self.Reward[t]-self.Q[self.A]) 
                self.RestA=range(self.n)
                self.p_a = self.softmax(self.H)                     # get the probability of each action
            #count reward                    
            self.RewardSum=np.add(self.RewardSum,self.Reward)
            
            if a%100==0:
                print("Constant StepSize Proceding: {}/{}".format(a,self.set)) 

        # set for plot                           
        self.RewardMean=np.divide(self.RewardSum,self.set)
        self.OptAction=np.divide(self.OptAction,self.set)


    def softmax(self,Ha):
        return np.exp(Ha)/sum(np.exp(Ha))
    
        

if __name__ =="__main__":
    SetA=200
    bandit=Bandit(set=SetA,StepSize=0.1) #greedy us
    bandit.RL()
    TimeStep=bandit.TimeStep
    RewardMean1=bandit.RewardMean
    OptAction1=bandit.OptAction
    bandit2=Bandit(set=SetA,StepSize=0.1, Baseline=False)
    bandit2.RL()
    RewardMean2=bandit2.RewardMean
    OptAction2=bandit2.OptAction
    # save to file 
    data=np.array([TimeStep,RewardMean1,OptAction1,RewardMean2,OptAction2])
    file=open("Chapt2.8_data","wb")
    pickle.dump(data,file)
    file.close()
    fig, axs=plt.subplots(2,1)
    axs[0].set_title("Gradient Bandit Programm with/without Baseline")
    axs[0].plot(range(TimeStep),RewardMean1,'b--',label="with BL")
    axs[0].plot(range(TimeStep),RewardMean2,'y-',label="without BL")
    # axs[0].set_xlabel("Time Steps")
    axs[0].set_ylabel("Average Reward")
    axs[0].legend(loc="lower right")
    axs[0].grid(axis='y',linestyle='--')

    axs[1].plot(range(TimeStep),OptAction1,'b--',label='with BL')
    axs[1].plot(range(TimeStep),OptAction2,'y-',label="without B")
    axs[1].set_xlabel("Time Steps")
    axs[1].set_ylabel("% Optimal Action")
    axs[1].grid(axis='y',linestyle='--')
    
    axs[1].legend(loc="lower right")
    axs[1].set_ylim(0,1)
    plt.show()
    fig.savefig("Chapt2.8.png")
