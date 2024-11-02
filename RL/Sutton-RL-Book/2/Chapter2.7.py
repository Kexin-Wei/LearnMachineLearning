# Created by Vac. 2020.8.3
# for Reinforcement Learning Book Chapter 2.7
# add UCB for sample average in 10 armed Bandit problem
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle

class Bandit:
    def __init__(self, BanditNum=10, q_mu=0., q_sigma=1.0, stationary=True, \
                    UpperConfidenceBound=True, c=2, \
                    Q0=5, TimeStep=1000, \
                    epsilon=0.1, set=600, StepSize=0.1):
        self.n = BanditNum                          # number of the Bandit
        self.Q = [ Q0 for i in range(self.n)]     # action value                
        self.N = np.zeros(BanditNum)                # record the times of choosing a

        self.set=set            # run time set (for average)
        self.stat = stationary  # if the bandit value change at every try
        self.epsilon = epsilon  # epsilon-greedy method for exploration
        self.q_mu = q_mu        # true value q*(a) ~ N(q_mu,q_sigma)
        self.q_sigma = q_sigma  

        self.Reward = np.zeros(TimeStep)    # 1000 record the reward for every try
        self.RewardSum = np.zeros(TimeStep) # sum the reward for each set
        self.StepSize = StepSize      # alpha parameter of target error a(R-Q)
        self.TimeStep = TimeStep    # running times in one set
        self.OptAction= np.zeros(TimeStep)

        self.UCB=UpperConfidenceBound       # if use upper confidence bound or not
        self.c=c                            # confidence level of Q

    def reset(self):
        self.q_n = np.random.normal( self.q_mu, self.q_sigma, self.n)
        self.Q=np.zeros(self.n)
        self.N=np.zeros(self.n)
    
    def RL(self, constF=True, aveF=False): #set up 2 Flags: 1 flag for constant step size, 1 for average

        for a in range(self.set):
        # start the time step
            self.reset()

            for t in range( self.TimeStep ):
            #True for stationary 
                if self.stat==False: 
                    self.q_n = np.random.normal( self.q_mu, self.q_sigma, self.n)
            # if in 1-e choose the best
                self.p = random.random()
                if self.p > self.epsilon: 
                    if aveF and self.UCB:
                        tempQ=np.zeros(self.n)
                        for i in range(self.n):
                            if self.N[i]:
                                tempQ[i] = self.Q[i]+self.c*np.sqrt(np.log(t)/self.N[i])
                            else:   # if Nta == 0 no +
                                tempQ[i] = self.Q[i] 
                        self.A = np.argmax(tempQ)
                    else:
                        self.A = np.argmax(self.Q)
            # else choose a random one                    
                else:                 
                    self.A = random.randrange(self.n)
                if self.A == np.argmax(self.q_n):
                    self.OptAction[t]+=1
                self.Reward[t] = self.q_n[self.A]
            #Constant StepSize
                if constF:
                    self.Q[self.A] +=self.StepSize*(self.Reward[t]-self.Q[self.A]) 
            # Sample Average
                if aveF:
                    self.N[self.A] +=1
                    self.Q[self.A] +=1.0/self.N[self.A]*(self.Reward[t]-self.Q[self.A])
            self.RewardSum=np.add(self.RewardSum,self.Reward)
            if aveF and a%100==0:
                print("Sample Average Proceding: {}/{}".format(a,self.set))
            if constF and a%100==0:
                print("Constant StepSize Proceding: {}/{}".format(a,self.set))                    
        self.RewardMean=np.divide(self.RewardSum,self.set)
        self.OptAction=np.divide(self.OptAction,self.set)

    def figTS(self):        
        plt.plot(range(self.TimeStep),self.RewardMean)
        plt.xlabel("Time Steps")
        plt.ylabel("Average Reward")

    def figOpt(self):      
        plt.plot(range(self.TimeStep),self.OptAction)
        plt.xlabel("Time Steps")
        plt.ylabel("% Optimal Action")
        

if __name__ =="__main__":
    bandit=Bandit(UpperConfidenceBound=False)
    bandit.RL(constF=False,aveF=True)
    TimeStep=bandit.TimeStep
    RewardMean1=bandit.RewardMean
    OptAction1=bandit.OptAction
    bandit2=Bandit(epsilon=0)
    bandit2.RL(constF=False,aveF=True)
    RewardMean2=bandit2.RewardMean
    OptAction2=bandit2.OptAction
    # save to file 
    data=np.array([TimeStep,RewardMean1,OptAction1,RewardMean2,OptAction2])
    file=open("Chapt2.7_data","wb")
    pickle.dump(data,file)
    file.close()
    fig, axs=plt.subplots(2,1)
    axs[0].set_title("Upper Confidence Bound Test")
    axs[0].plot(range(TimeStep),RewardMean1,'b--',label="no UCB")
    axs[0].plot(range(TimeStep),RewardMean2,'y-',label="with UCB")
    # axs[0].set_xlabel("Time Steps")
    axs[0].set_ylabel("Average Reward")
    axs[0].legend(loc="lower right")
    axs[0].grid(axis='y',linestyle='--')

    axs[1].plot(range(TimeStep),OptAction1,'b--',label='no UCB')
    axs[1].plot(range(TimeStep),OptAction2,'y-',label="with UCB")
    axs[1].set_xlabel("Time Steps")
    axs[1].set_ylabel("% Optimal Action")
    axs[1].grid(axis='y',linestyle='--')
    
    axs[1].legend(loc="lower right")
    axs[1].set_ylim(0,1)
    plt.show()
    fig.savefig("Chapt2.7.png")
