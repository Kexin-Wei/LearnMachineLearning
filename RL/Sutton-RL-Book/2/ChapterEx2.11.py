# Created by Vac. 2020.8.4
# for Reinforcement Learning Book Chapter 2 Ex 2.11
# compare epsilon-greedy, gradient bandit, UCB and optimistic initialization with constant step
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle

class Bandit:
    def __init__(self, BanditNum=10, q_mu=0., q_sigma=1.0, stationary=True, 
                    epsilon=0.1,                         
                    c=2,     
                    Q0=0.0, StepSize=0.1,                 
                    BaseLine=True,
                    Amethod="epsilong",Qmethod="ave",
                    TimeStep=1000, set=2000):           
        """
        BanditNum=10, q_mu=0., q_sigma=1.0, stationary=True, # bandit set up
        epsilon=0.1,                         # epsilon- greedy
        c=2,                                 # UCB, confidence level
        Q0=0, StepSize=0.1,                  # optimistic initialization
        BaseLine=True,                       # Gradient Bandit
        TimeStep=1000, set=2000):            # trainning set up
        Amethod="epsilong",Qmethod="ave"     # different option for A and Q, see below
        """
        self.n = BanditNum                  # number of the Bandit
        self.q_mu = q_mu                    # true value q*(a) ~ N(q_mu,q_sigma)
        self.q_sigma = q_sigma
        self.stat = stationary              # if the bandit value change at every try
        self.Q0=Q0

        self.Reward = np.zeros(TimeStep)    # 1000 record the reward for every try
        self.RewardSum = np.zeros(TimeStep) # sum the reward for each set
        self.OptAction= np.zeros(TimeStep)

        self.N = np.zeros(self.n)           # record the times of choosing a
        self.epsilon = epsilon              # epsilon-greedy method for exploration

        self.c=c                            # confidence level of Q

        self.StepSize = StepSize            # alpha parameter of target error a(R-Q)

        self.Q = np.array( [ self.Q0 for i in range(self.n)] )  
                                            # action value initilization

        # self.H = np.zeros(self.n)           # the action preference H for Gradient Bandit Program 
        # use self.Q
        self.p_a = np.ones(self.n)/self.n   # the policy function in soft max distribution
        self.RewardAve = 0              # the Baseline of the Gradient Bandit Programm
        self.RestA=np.arange(self.n)        # prepare for action update
        self.baselineF=BaseLine         # use baseline or not

        self.set=set                # run time set (for average)
        self.TimeStep = TimeStep    # running times in one set
        self.Amethod=Amethod        # Amethop Option
                                        # epsilong:   epsilon-greedy
                                        # UCB:        upper confidence bound
                                        # gradient:   gradient bandit programm
        self.Qmethod=Qmethod        # Qmethod Option
                                        # const: constant step size
                                        # ave:   average sample
        if self.Amethod=="UCB": self.Qmethod="ave"
        if self.Amethod=="gradient": self.Qmethod="gradient"

    def reset(self):
        self.q_n = np.random.normal( self.q_mu, self.q_sigma, self.n)
        self.Q =np.array( [ self.Q0 for i in range(self.n)] )
        self.N=np.zeros(self.n)

        self.RestA=np.arange(self.n)
        self.RewardAve = 0
        self.p_a = np.ones(self.n)/self.n 
    
    def qStation(self): #if non stationary, change everytime
        if self.stat==False: 
            self.q_n = np.random.normal( self.q_mu, self.q_sigma, self.n)

    def isOptAct(self,t): #if this A is the best action
        if self.A == np.argmax(self.q_n):
            self.OptAction[t]+=1

    def softmax(self,Ha):
        return np.exp(Ha)/sum(np.exp(Ha))
    
    def Acal(self,t):
        if self.Amethod=="epsilong": #epsilon-greedy
            if random.random() > self.epsilon: # if in 1-e choose the best
                return argmax(self.Q)     
            else:
                return np.random.choice(np.arange(self.n)) # else choose a random one 

        if self.Amethod=="UCB":
            return argmax(self.Q+self.c*np.sqrt(np.log(t+1)/(self.N+1e-5)))

        if self.Amethod=="gradient":
            return np.random.choice(np.arange(self.n),p=self.p_a)

    def Qcal(self,t):      
        if self.Qmethod=="const":
            self.Q[self.A] +=self.StepSize*(self.Reward[t]-self.Q[self.A])

        if self.Qmethod=="ave":
            self.N[self.A] +=1
            self.Q[self.A] +=1.0/self.N[self.A]*(self.Reward[t]-self.Q[self.A])
        
        if self.Qmethod=="gradient":
            if self.baselineF: # use baseline for probability cal
                self.RewardAve = self.RewardAve+(self.Reward[t]-self.RewardAve)/(t+1)   # cal the average reward
            else:
                self.RewardAve=0
            self.RestA=np.delete(self.RestA,self.A) #get the a!=A
            self.Q[self.A]=self.Q[self.A]+self.StepSize*(self.Reward[t]-self.RewardAve)*(1-self.p_a[self.A])
            self.Q[self.RestA]=self.Q[self.RestA]-self.StepSize*(self.Reward[t]-self.RewardAve)*self.p_a[self.RestA]
            self.RestA=np.arange(self.n)
            self.p_a = self.softmax(self.Q)                     # get the probability of each action

    def prtIteration(self,a):
        if a%100==0:
            if self.Qmethod=="const":
                print("Constant StepSize Proceding: {}/{}".format(a,self.set))                    
            if self.Qmethod=="ave":
                print("Sample Average Proceding: {}/{}".format(a,self.set))
            if self.Qmethod=="gradient":
                print("Gradient Bandit Proceding: {}/{}".format(a,self.set))

    def RL(self):    #True for Constant Step Size, False for average sample
        for a in range(self.set):
            # start the time step
            self.reset()
            for t in range( self.TimeStep ):
                self.qStation()

                self.A=self.Acal(t)
                
                self.isOptAct(t)

                self.Reward[t] = self.q_n[self.A]

                self.Qcal(t)

            self.prtIteration(a)         
            self.RewardSum=np.add(self.RewardSum,self.Reward)
        print("end.")
        self.RewardMean=np.divide(self.RewardSum,self.set)
        self.OptAction=np.divide(self.OptAction,self.set)
    

def argmax(q):
    maxlist=[]
    max=0
    for i in range(len(q)):
        if q[i] > max:
            max=q[i]
            maxlist=[]
        if q[i] == max:
            maxlist.append(i)
    return  np.random.choice(maxlist)

def figGrad(mu,iteration):
    bandit=Bandit(q_mu=mu,Amethod="gradient",set=iteration)
    bandit.RL()
    bandit2=Bandit(q_mu=mu,Amethod="gradient",BaseLine=False,set=iteration)
    bandit2.RL()
    data=np.array([bandit.TimeStep,bandit.RewardMean,bandit.OptAction,bandit2.RewardMean,bandit2.OptAction])

    file=open("Chapt2Ex2.11_Gradient","wb")
    pickle.dump(data,file)
    file.close()

    string1="with Baseline"
    string2="without baseline"
    fig, axs=plt.subplots(2,1)
    axs[0].set_title("Gradient Bandit Programm")
    axs[0].plot(range(bandit.TimeStep),bandit.RewardMean,'b--',label=string1)
    axs[0].plot(range(bandit.TimeStep),bandit2.RewardMean,'y-',label=string2)
    # axs[0].set_xlabel("Time Steps")
    axs[0].set_ylabel("Average Reward")
    axs[0].legend(loc="lower right")
    axs[0].grid(axis='y',linestyle='--')

    axs[1].plot(range(bandit.TimeStep),bandit.OptAction,'b--',label=string1)
    axs[1].plot(range(bandit.TimeStep),bandit2.OptAction,'y-',label=string2)
    axs[1].set_xlabel("Time Steps")
    axs[1].set_ylabel("% Optimal Action")
    axs[1].grid(axis='y',linestyle='--')
    axs[1].legend(loc="lower right")
    axs[1].set_ylim(0,1)
    plt.show()
    fig.savefig("Chapt2Ex2.11_Gradient.png")

def figUCB(c,epsilon2,iteration):
    bandit=Bandit(c=2,epsilon=0,Amethod="UCB",set=iteration)
    bandit.RL()

    bandit2=Bandit(epsilon=epsilon2,set=iteration)
    bandit2.RL()

    data=np.array([bandit.TimeStep,bandit.RewardMean,bandit.OptAction,bandit2.RewardMean,bandit2.OptAction])
    file=open("Chapt2Ex2.11_UCB","wb")
    pickle.dump(data,file)
    file.close()

    string1='UCB,c = '+str(c)
    string2='epsilon = '+str(epsilon2)
    fig, axs=plt.subplots(2,1)
    axs[0].set_title("UCB vs Epsilon-Greedy")
    axs[0].plot(range(bandit.TimeStep),bandit.RewardMean,'b--',label=string1)
    axs[0].plot(range(bandit.TimeStep),bandit2.RewardMean,'y-',label=string2)
    # axs[0].set_xlabel("Time Steps")
    axs[0].set_ylabel("Average Reward")
    axs[0].legend(loc="lower right")
    axs[0].grid(axis='y',linestyle='--')

    axs[1].plot(range(bandit.TimeStep),bandit.OptAction,'b--',label=string1)
    axs[1].plot(range(bandit.TimeStep),bandit2.OptAction,'y-',label=string2)
    axs[1].set_xlabel("Time Steps")
    axs[1].set_ylabel("% Optimal Action")
    axs[1].grid(axis='y',linestyle='--')
    axs[1].legend(loc="lower right")
    axs[1].set_ylim(0,1)
    plt.show()
    fig.savefig("Chapt2Ex2.11_UCB.png")

def figEpsilonGreedy(epsilon1,epsilon2,iteration,Am,Qm):
    bandit=Bandit(epsilon=epsilon1,set=iteration,Amethod=Am,Qmethod=Qm)
    bandit.RL()
    
    bandit2=Bandit(epsilon=epsilon2,set=iteration,Amethod=Am,Qmethod=Qm)
    bandit2.RL()

    # save to file 
    data=np.array([bandit.TimeStep,bandit.RewardMean,bandit.OptAction,bandit2.RewardMean,bandit2.OptAction])
    file=open("Chapt2Ex2.11_epsilon","wb")
    pickle.dump(data,file)
    file.close()
    
    string1='epsilon = '+str(epsilon1)
    string2='epsilon = '+str(epsilon2)
    fig, axs=plt.subplots(2,1)
    axs[0].set_title("Epsilon-Greedy Method with Parameter Comparation")
    axs[0].plot(range(bandit.TimeStep),bandit.RewardMean,'b--',label=string1)
    axs[0].plot(range(bandit.TimeStep),bandit2.RewardMean,'y-',label=string2)
    # axs[0].set_xlabel("Time Steps")
    axs[0].set_ylabel("Average Reward")
    axs[0].legend(loc="lower right")
    axs[0].grid(axis='y',linestyle='--')

    axs[1].plot(range(bandit.TimeStep),bandit.OptAction,'b--',label=string1)
    axs[1].plot(range(bandit.TimeStep),bandit2.OptAction,'y-',label=string2)
    axs[1].set_xlabel("Time Steps")
    axs[1].set_ylabel("% Optimal Action")
    axs[1].grid(axis='y',linestyle='--')
    axs[1].legend(loc="lower right")
    axs[1].set_ylim(0,1)
    plt.show()
    fig.savefig("Chapt2Ex2.11_epsilon.png")

def figOptmInit(optminit,epsilon2,iteration,Am,Qm):
    bandit=Bandit(Q0=optminit,epsilon=0,set=iteration,Amethod=Am,Qmethod=Qm)
    bandit.RL()

    bandit2=Bandit(epsilon=epsilon2,set=iteration,Amethod=Am,Qmethod=Qm)
    bandit2.RL()   
    # save to file 
    data=np.array([bandit.TimeStep,bandit.RewardMean,bandit.OptAction,bandit2.RewardMean,bandit2.OptAction])
    file=open("Chapt2Ex2.11_OptmInit","wb")
    pickle.dump(data,file)
    file.close()
    
    string1='Q1 = '+str(optminit)+', epsilon = 0'
    string2='Q1 = 0, epsilon = '+str(epsilon2)
    fig, axs=plt.subplots(2,1)
    axs[0].set_title("Optimistic Initilization")
    axs[0].plot(range(bandit.TimeStep),bandit.RewardMean,'b--',label=string1)
    axs[0].plot(range(bandit.TimeStep),bandit2.RewardMean,'y-',label=string2)
    # axs[0].set_xlabel("Time Steps")
    axs[0].set_ylabel("Average Reward")
    axs[0].legend(loc="lower right")
    axs[0].grid(axis='y',linestyle='--')

    axs[1].plot(range(bandit.TimeStep),bandit.OptAction,'b--',label=string1)
    axs[1].plot(range(bandit.TimeStep),bandit2.OptAction,'y-',label=string2)
    axs[1].set_xlabel("Time Steps")
    axs[1].set_ylabel("% Optimal Action")
    axs[1].grid(axis='y',linestyle='--')
    axs[1].legend(loc="lower right")
    axs[1].set_ylim(0,1)
    plt.show()
    fig.savefig("Chapt2Ex2.11_OptmInit.png")



if __name__ =="__main__":
    # figEpsilonGreedy(0.1,0.01,200,"epsilong","ave")
    # figOptmInit(5,0.1,600,"epsilong","const")
    figUCB(2,0.1,200)
    # figGrad(4.0,200)