# Created by Vac. 2020.7.21
# for Reinforcement Learning Book Chapter 2 Excercise 2.5
# compare sample average and constant step size with 10 armed Bandit problem
import numpy as np
import random
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, BanditNum=10, q_mu=0., q_sigma=1.0, stationary=True, \
                    Q0=[[5 for i in range(10)],[5 for i in range(10)] ], TimeStep=1000, \
                    epsilon=0.1, set=200, StepSize=0.1):
        self.n = BanditNum
        self.q_mu = q_mu
        self.q_sigma = q_sigma
        self.stat = stationary
        self.Q = Q0
        self.N = np.zeros(BanditNum)
        self.TimeStep = TimeStep
        self.epsilon = epsilon
        self.Reward = np.zeros([2,TimeStep]) # 1000
        self.RewardSum = np.zeros([2,TimeStep]) # 1000
        self.set=set
        self.OptAction= np.zeros([2,TimeStep])
        self.StepSize=StepSize
        self.A=0

    def reset(self):
        self.q_n = np.random.normal( self.q_mu, self.q_sigma, self.n)
        self.Q=np.zeros([2,self.n])
        self.N=np.zeros(self.n)

    def RL(self):
        for a in range(self.set):
        # start the time step
            self.reset()
            for t in range( self.TimeStep ):
                if self.stat==False: #True for stationary 
                    self.q_n = np.random.normal( self.q_mu, self.q_sigma, self.n)
                self.p = random.random()
                if self.p > self.epsilon: # if in 1-e choose the best
                    self.A0 = np.argmax(self.Q[0,:])
                    self.A1 = np.argmax(self.Q[1,:])
                else:                  # else choose a random one
                    self.A0 = random.randrange(self.n)
                    self.A1=self.A0
                if self.A0 == np.argmax(self.q_n):
                    self.OptAction[0,t]+=1
                if self.A1 == np.argmax(self.q_n):
                    self.OptAction[1,t]+=1
                self.Reward[0,t] = self.q_n[self.A0]
                self.Reward[1,t] = self.q_n[self.A1]
                self.N[self.A1] +=1
                self.Q[0,self.A0] +=self.StepSize*(self.Reward[0,t]-self.Q[0,self.A0]) #Constant StepSize
                self.Q[1,self.A1] +=1.0/self.N[self.A1]*(self.Reward[1,t]-self.Q[1,self.A1])# Sample Average
            self.RewardSum[0,:]=np.add(self.RewardSum[0,:],self.Reward[0,:])
            self.RewardSum[1,:]=np.add(self.RewardSum[1,:],self.Reward[1,:])
            if a%100==0:
                print("Proceding: {}/{}".format(a,self.set))
        self.RewardMean=np.divide(self.RewardSum,self.set)
        self.OptAction=np.divide(self.OptAction,self.set)

    def figAveReward(self):
        ave=plt.figure()
        plt.plot(range(self.TimeStep),self.RewardMean[0,:],'b--',label='Constant Step Size')
        plt.plot(range(self.TimeStep),self.RewardMean[1,:],'r-',label='Sample Average')
        plt.xlabel("Time Steps")
        plt.ylabel("Average Reward")
        plt.legend(loc="lower right")
        ave.savefig("Chapt2Ex2.5_AveReward.png")

    def figOpAct(self):
        opt=plt.figure()
        plt.plot(range(self.TimeStep),self.OptAction[0,:],'b--',label='Constant Step Size')
        plt.plot(range(self.TimeStep),self.OptAction[1,:],'r-',label='Sample Average')
        plt.xlabel("Time Steps")
        plt.ylabel("% Optimal Action")
        plt.legend(loc="lower right")
        opt.savefig("Chapt2Ex2.5_OptAct.png")
        plt.show()

if __name__ =="__main__":
    bandit=Bandit()
    bandit.RL()
    bandit.figAveReward()
    bandit.figOpAct()

