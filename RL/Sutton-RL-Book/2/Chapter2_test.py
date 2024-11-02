# Created by Vac. 2020.7.21
# for Reinforcement Learning Book Chapter 2 Excercise 2.5
# constant step size vs average sample
import numpy as np
import random
import matplotlib.pyplot as plt
BanditNum=10
q_mu=0.
q_sigma=1.0
stationary=False
Q0=np.zeros([2,10])
TimeStep=1000
epsilon=0.1
set=2000
StepSize=0.1


n = BanditNum
q_mu = q_mu
q_sigma = q_sigma
stat = stationary
Q = Q0
N = np.zeros(10)
TimeStep = TimeStep
epsilon = epsilon
Reward = np.zeros([2,TimeStep]) # 1000
RewardSum = np.zeros([2,TimeStep]) # 1000
set=set
OptAction= np.zeros([2,TimeStep])
StepSize=StepSize
A=0
# sample average
for a in range(set):
    q_n = np.random.normal( q_mu, q_sigma, n)
    Q=np.zeros([2,10])
    N=np.zeros(10)
    # start the time step
    for t in range( TimeStep ):
        q_n = np.random.normal( q_mu, q_sigma, n)
        p = random.random()
        if p > epsilon: # if in 1-e choose the best
            A0 = np.argmax(Q[0,:])
            A1 = np.argmax(Q[1,:])
        else:                  # else choose a random one
            A0 = random.randrange(n)
            A1=A0
        if A0 == np.argmax(q_n):
            OptAction[0,t]+=1
        if A1 == np.argmax(q_n):
            OptAction[1,t]+=1
        Reward[0,t] = q_n[A0]
        Reward[1,t] = q_n[A1]
        N[A1] +=1
        Q[0,A0] +=StepSize*(Reward[0,t]-Q[0,A0]) #Constant StepSize
        Q[1,A1] +=1.0/N[A1]*(Reward[1,t]-Q[1,A1])# Sample Average
    RewardSum[0,:]=np.add(RewardSum[0,:],Reward[0,:])
    RewardSum[1,:]=np.add(RewardSum[1,:],Reward[1,:])
    if a%50==0:
        print("Proceding: {}/{}".format(a,set))
RewardMean=np.divide(RewardSum,set)
OptAction=np.divide(OptAction,set)

plt.figure()
plt.plot(range(TimeStep),RewardMean[0,:],'b--',label='Constant Step Size')
plt.plot(range(TimeStep),RewardMean[1,:],'r-',label='Sample Average')
plt.xlabel("Time Steps")
plt.ylabel("Average Reward")
plt.legend(loc="lower right")

plt.figure()
plt.plot(range(TimeStep),OptAction[0,:],'b--',label='Constant Step Size')
plt.plot(range(TimeStep),OptAction[1,:],'r-',label='Sample Average')
plt.xlabel("Time Steps")
plt.ylabel("% Optimal Action")
plt.legend(loc="lower right")
plt.show()