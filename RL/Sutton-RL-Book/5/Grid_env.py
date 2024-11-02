"""
Grid 4x4:
- terminal: (0,3)
- policy: random 0~1
- value: all zero
- reward: each step -1
- statemove: define by StateMove()
"""
import numpy as np
import matplotlib.pyplot as plt

class Grid:
    def __init__(self,gamma=0.9,Terminal=[[0,3]],policy='random',epsilon=0.3,qvalue='random',grid_dim=[4,4]):
        self.Policy = np.zeros((4,4,4))
        # default random 
        if policy=='random':    # initial a random policy
            for i in range(4):
                for j in range(4):
                    p=np.random.rand(4)
                    p=p/sum(p)
                    self.Policy[i,j]=p

        elif policy=='ep_soft':  # initial a epsilon-soft policy
            for i in range(4):
                for j in range(4):
                    p=np.random.rand(4)+0.25
                    p=p/sum(p)
                    self.Policy[i,j]=p

        elif policy=='equal':   # initial a equal probability for actions
            self.Policy=np.ones((4,4,4))*0.25

        else:
            self.Policy=policy

         
        self.Value = np.zeros((4,4)) # 0 for all state value
        self.gamma=gamma
        self.Terminal=Terminal
        self.epsilon=epsilon
        
        if qvalue=='zero':
            self.QValue=np.zeros((4,4,4))
        elif qvalue=='random':
            self.QValue=-np.random.randint(10,size=(4,4,4))
        else:
            self.QValue=qvalue            
        

    def IsTerminal(self,state): # define when to end this episode
        # use T determinate the item in Terminal
        # [0,1] is equal with i,j
        if len(self.Terminal)>1:
            for T in self.Terminal:
                if state == T:
                    return True  
        if state==self.Terminal[0]:
            return True
        return False

    def Action(self,state): # return the action respect to the policy at state(i,j)                
        return np.random.choice(4,p=self.Policy[state[0],state[1]])

    def getReward(self,state,next_state): #each step cost
        return -1

    def NextState(self,action,state): #define the move rules
        # return the next move of i,j
        # action 0 -> left
        # action 1 -> up
        # action 2 -> right
        # action 3 -> down
        # if x
        x=0
        y=0
        i,j=state
        if action==0: #left
            x=i-1
            y=j   
        if action==1: #up
            x=i
            y=j+1
        if action==2: #right
            x=i+1
            y=j
        if action==3: #down
            x=i
            y=j-1
        if x<0: x=0
        if x>3: x=3
        if y<0: y=0
        if y>3: y=3
        return [x, y]

    def PolicyUpdate(self,state,update_op='greedy'):   
        Q_s_a=self.QValue[state[0],state[1]]
        # find the max, when the rest is smaller, put the rest to 0
        max_2 = round(max(Q_s_a), 2)
        a_range = len(Q_s_a)
        candidate = [] # randomly choose a max one *accuracy:+-0.01
        for a in range(a_range):
            if round(Q_s_a[a], 2) == max_2: candidate.append(a)
        theone=np.random.choice(candidate) # the argmax_a

        # 2 options:
        # 1. epsilon-soft: keeping exploring
        if update_op=='ep_soft':
            pi_new=np.ones(4)*self.epsilon/4
            pi_new[theone]+=1-self.epsilon
        # 2. greedy choose optimal at each steps         
        else:
            pi_new=np.zeros(4)
            pi_new[theone]=1
        return pi_new

    def figPlotValue(self,name,value="qv"):
        fig=plt.figure(figsize=(4,4))
        plt.xlim([0,4])
        plt.ylim([0,4])
        ax=fig.gca()
        ax.set_xticks(np.arange(0,5,1))
        ax.set_yticks(np.arange(0,5,1))
        plt.grid()  #set grid

        #ax = fig.add_subplot(111)
        #plt.gca().add_patch(plt.Rectangle((0,3),1,1,color="black"))
        #plt.gca().add_patch(plt.Rectangle((3,0),1,1,color="black"))# set rectangle

        # plot value options:
        # 1.Qvalue(state,action):   qv
        # 2.Policy:                 p
        # 3.Value(state):           v
        if value=='p':
            V=self.Policy
        elif value == 'v':
            V=self.Value
        else:
            V=self.QValue

        q= len(V.shape)==3 # if V is QValue or 3 dimensions
        for i in range(4):
            for j in range(4):
                #if self.V[i][j]!=0
                if q:
                    plt.text(i+0.1,j+0.4,  round(V[i,j,0],2),fontsize=8) #left
                    plt.text(i+0.35,j+0.65,round(V[i,j,1],2),fontsize=8) #up
                    plt.text(i+0.6,j+0.4,  round(V[i,j,2],2),fontsize=8) #right                    
                    plt.text(i+0.35,j+0.15,round(V[i,j,3],2),fontsize=8) #down
                else:
                    plt.text(i+0.2,j+0.35,round(V[i,j],2))
        plt.savefig(name)
        plt.close()

    def figPlotMove(self,name):
        fig=plt.figure(figsize=(4,4))
        plt.xlim([0,4])
        plt.ylim([0,4])
        ax=fig.gca()
        ax.set_xticks(np.arange(0,5,1))
        ax.set_yticks(np.arange(0,5,1))
        plt.grid()  #set grid

        ax = fig.add_subplot(111)
        for T in self.Terminal:
            plt.gca().add_patch(plt.Rectangle((T[0],T[1]),1,1,color="black"))
        

        for i in range(4):
            for j in range(4):
                if self.Policy[i,j,0]!=0: plt.annotate(text="",xy=(i,j+0.5),xytext=(i+0.5,j+0.5),arrowprops={"arrowstyle":"->"}) #left
                if self.Policy[i,j,1]!=0: plt.annotate(text="",xy=(i+0.5,j+1),xytext=(i+0.5,j+0.5),arrowprops={"arrowstyle":"->"}) #up
                if self.Policy[i,j,2]!=0: plt.annotate(text="",xy=(i+1,j+0.5),xytext=(i+0.5,j+0.5),arrowprops={"arrowstyle":"->"}) #right
                if self.Policy[i,j,3]!=0: plt.annotate(text="",xy=(i+0.5,j),xytext=(i+0.5,j+0.5),arrowprops={"arrowstyle":"->"}) #down                          
        plt.savefig(name)
        plt.close()