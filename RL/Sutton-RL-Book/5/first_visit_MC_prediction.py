# on policy
from Grid_env import Grid
import numpy as np
import matplotlib.pyplot as plt


class Grid_MC_P(Grid):
    def __init__(self, gamma=0.9,Terminal=[0,3],**kwargs):
        super().__init__(gamma,Terminal)
        self.Return_n=np.zeros((4,4))
        self.steps=0
        for key,value in kwargs.items():  # accept self defined policy for prediction
            if key == 'Policy':
                self.Policy=value
    
    def run_one_episode(self,start_state=[0,0]):
        state=start_state   # fixed for prediction 
        trajectory_p=[state]
        trajectory_a=[]
        self.steps=0
        while(1):            
            action=self.Action(state)
            trajectory_a.append(action)
            next_s=self.NextState(action,state)
            if self.IsTerminal(next_s):
                #print('End with',self.steps,'Steps')
                break
            trajectory_p.append(next_s)
            state=next_s
            self.steps+=1
            #if self.steps%20==0:
            #    print('Step:',self.steps,'\tState: (',state[0],',',state[1],')')
            
        return trajectory_p,trajectory_a
    
    def value_update(self):
        trajectory_p,trajectory_a=self.run_one_episode()
        G=0
        for i in range(1,self.steps): # i=1,2,...T-1
            t=-i-1
            rewards=self.getReward(trajectory_p[t],trajectory_p[t+1])
            G=self.gamma*G+rewards
            S_t=trajectory_p[t]
            if S_t not in trajectory_p[:(t-1)]:
                self.Return_n[S_t[0],S_t[1]]+=1
                v=self.Value[S_t[0],S_t[1]]
                self.Value[S_t[0],S_t[1]]=v+(G-v)/self.Return_n[S_t[0],S_t[1]]
    
    def run_n_episode(self,n_ep):
        run_time=0
        while(run_time<n_ep):
            self.value_update()
            run_time+=1
            if run_time%20==0:
                print('Done',run_time,'Episodes..')
        

def main():
    policy=np.ones((4,4,4))*0.25
    agent=Grid_MC_P(Policy=policy,Terminal=[[3,0]])
    agent.run_n_episode(1000)
    agent.figPlotValue('5/On PolicyFirst MC Prediction',value='v')


if __name__=="__main__":
    main()