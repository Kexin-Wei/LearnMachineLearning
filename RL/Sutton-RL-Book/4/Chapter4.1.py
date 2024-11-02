"""
created by Vac, 2020.8.11
reshown the figure 4.1 from RL book
Policy Iteration for policy optimalization

Algorithem:

1. Initialization
    V(s) in R and pi(s) in A(s) arbitrarily for all s in S

2. Policy Evaluation
    Loop:
        delta<-0
        for each s in S:
            v<-V(s)
            V(s)<-sum(pi(s)*sum(p(s',r|s,a)*(r + gamma* V (s')))
            delta<-max( delta , |v-V(s)|)
    until  delta < theta (a small positive number determining the accuracy of estimation)

3. Policy Improvement
    policy-stable<-true
    For each s in S:
        old-action<-pi(s)
        pi(s)<-argmaxa(P(s',r|s,a)*(r +  gamma*V (s')))
    If old-action !=pi(s), then policy-stable<-false
    If policy-stable, then stop and return V as v_opt and Pi as pi_opt; else go to 2
"""
import numpy as np
import matplotlib.pyplot as plt

class Grid:
    def __init__(self,theta=0.03,gamma=0.9):
        self.pi_s = [0.25, 0.25, 0.25, 0.25]
        self.Pi = [[self.pi_s for i in range(4)] for i in range(4)]
        self.V = [[0 for i in range(4)]for i in range(4)]
        self.Terminal=[[0,3],[3,0]]
        self.delta=0
        self.theta=theta
        self.gamma=gamma
        self.ItFlag=False
        self.runtime=0

    def isTerminal(self,i,j):
        # use T determinate the item in Terminal
        # [0,1] is equal with i,j
        for T in self.Terminal:
            if i == T[0] and j ==T[1]:
                return True
        
        return False
                
    def StateMove(self,move,i,j):
        # return the next move of i,j
        # move 0 -> left
        # move 1 -> up
        
        # move 2 -> right
        # move 3 -> down
        # if x
        x=0
        y=0
        if move==0: #left
            x=i-1
            y=j   
        if move==1: #up
            x=i
            y=j+1
        if move==2: #right
            x=i+1
            y=j
        if move==3: #down
            x=i
            y=j-1
        if x<0: x=0
        if x>3: x=3
        if y<0: y=0
        if y>3: y=3
        return x, y

    def getReward(self,x,y,x2,y2):
        return -1

    def argmax_a(self,choice,Pi):
        # find the max, when the rest is smaller, put it to 0
        max_expection=round(max(choice),3) #max float 0.001
        pi_new=[0.0, 0.0, 0.0, 0.0]
        for a in range(4):
            if round(choice[a],3) == max_expection: pi_new[a]=Pi[a]
            else: pi_new[a]=0
        return pi_new


    def figPlotValue(self,name):
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

        for i in range(4):
            for j in range(4):
                #if self.V[i][j]!=0:
                    plt.text(i+0.2,j+0.35,round(self.V[i][j],2))
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
        plt.gca().add_patch(plt.Rectangle((0,3),1,1,color="black"))
        plt.gca().add_patch(plt.Rectangle((3,0),1,1,color="black"))# set rectangle

        for i in range(4):
            for j in range(4):
                for k in range(4):
                    if self.Pi[i][j][k]!=0:
                        if k==0: plt.annotate(text="",xy=(i,j+0.5),xytext=(i+0.5,j+0.5),arrowprops={"arrowstyle":"->"}) #left
                        if k==1: plt.annotate(text="",xy=(i+0.5,j+1),xytext=(i+0.5,j+0.5),arrowprops={"arrowstyle":"->"}) #up
                        if k==2: plt.annotate(text="",xy=(i+1,j+0.5),xytext=(i+0.5,j+0.5),arrowprops={"arrowstyle":"->"}) #right
                        if k==3: plt.annotate(text="",xy=(i+0.5,j),xytext=(i+0.5,j+0.5),arrowprops={"arrowstyle":"->"}) #down
        plt.savefig(name)
        plt.close()


    def PolicyEvaluation(self):
        iteration=0
        while(self.delta>self.theta or iteration==0):   #if one of the Vs is unstable, continue
            self.delta=0                                            # the first time delta ==0, push into while
            iteration+=1
            print('PolicyEvaluation - Iteration: ',iteration)
            for i in range(4):
                for j in range(4):
                    v=self.V[i][j]
                    
                    new_value=0 #calculate value
                    for a in range(4):
                        x_next,y_next=self.StateMove(a,i,j)
                        self.reward=self.getReward(i,j,x_next,y_next)
                        new_value+=self.Pi[i][j][a]*(self.reward+self.gamma*self.V[x_next][y_next])
                    
                    if self.isTerminal(i,j): # exclude the terminal
                        self.V[i][j]=0
                    else:
                        self.V[i][j]=new_value

                    self.delta=max(self.delta,abs(v-self.V[i][j])) # check if the Vs is stable, delta is the biggest value in |v-Vs|
                    #print('PolicyEvaluatino - Iteration: ',iteration,' delta:', self.delta)
            valuename='fig4.1/RunTime'+str(self.runtime)+'_Iteration'+str(iteration)+'.png'
            self.figPlotValue(valuename)       
            

    def PolicyIteration(self):
        
        Pi_old=self.Pi
        movetime=0

        for i in range(4):
            for j in range(4):

                # pi(s)<-argmax_a sum_s'_r(p*(r+gamma*V(s')))  
                choice=[]
                for a in range(4):
                    x_next,y_next=self.StateMove(a,i,j) #get the next i,j position
                    self.reward=self.getReward(i,j,x_next,y_next) # get the next reward
                    choice.append(self.Pi[i][j][a]*(self.reward+self.gamma*self.V[x_next][y_next]))
                
                self.Pi[i][j]=self.argmax_a(choice,self.Pi[i][j])
                
                print('PolicyIteration...')
                movename='fig4.1/RunTime'+str(self.runtime)+'_Move'+str(movetime)+'.png'
                self.figPlotMove(name=movename)
                movetime+=1

                if Pi_old[i][j]!=self.Pi[i][j]:
                    return False
        return True



    def RL(self):
        plt.close('all')
        while(self.ItFlag==False):
            self.PolicyEvaluation()
            self.ItFlag=self.PolicyIteration()  

            self.runtime+=1
        print('Done. Total runtime:',self.runtime)

     
if __name__ == "__main__":
    grid=Grid(gamma=1)
    grid.RL()