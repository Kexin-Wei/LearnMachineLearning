from Grid_env import Grid
import numpy as np
import os
import time

class Grid_MC_Off_Pre():
    def __init__(self, gamma=0.9,Terminal=[[0,3]],
                       b_policy='random',t_policy='equal',**kwargs): 
                       # policy options see Grid_env.py
        self.b_agent=Grid(gamma,Terminal,b_policy)
        
        t_policy=self.initial_targert_agent()
        self.t_agent=Grid(gamma,Terminal,t_policy)
        self.C=np.zeros((4,4,4)) # index: state_0,state_1,action
        self.steps=0
        
    def initial_targert_agent(self):
        t_policy=np.zeros((4,4,4))
        for i in range(4):
            for j in range(4):
                p=[0]
                while(not sum(p)):
                    p=np.random.randint(2,size=(4))/2
                p=p/sum(p)
                t_policy[i,j]=p
        return t_policy


    def initial_behaviour_agent(self):
        for i in range(4):
            for j in range(4):
                non_zero_index=np.nonzero(self.t_agent.Policy[i,j])
                off_set=[ 1 if k in non_zero_index[0] else 0 for k in range(4)]
                p=np.random.rand(4)+np.array(off_set)
                p=p/sum(p)
                self.b_agent.Policy[i,j]=p
    
    def explore_start(self): # no use for this func
        while(1):
            s_0=np.random.randint(4,size=2).tolist()
            a_0=np.random.randint(4,size=1)[0]
            p=self.b_agent.Policy[s_0[0],s_0[1],a_0]
            if p > 0:
                return s_0,a_0

    def run_one_episode(self,start_state=[0,0],start_action=0):

        self.initial_behaviour_agent()

        trajectory={'state':[],
                    'action':[],
                    'reward':[]}

        state=start_state
        action=start_action

        self.steps=0
        while(1):                                               
            next_s=self.b_agent.NextState(action,state)
            reward=self.b_agent.getReward(state,next_s)

            trajectory['state'].append(state)
            trajectory['action'].append(action) 
            trajectory['reward'].append(reward)

            if self.b_agent.IsTerminal(next_s):                
                break
            state=next_s
            action=self.b_agent.Action(state)
            self.steps+=1             

        #print(f'End with {self.steps} Steps')    
        return trajectory
    
    def value_update(self,start_op='random'):
        if start_op=='random':
            s_0,a_0=self.explore_start()
        else:
            s_0=start_op['state']            
            a_0=start_op['action']

        trajectory=self.run_one_episode(start_action=a_0,start_state=s_0)

        G=0
        W=1
        for i in range(1,self.steps): # i=0,1,2,...T-1
            t=-i
            
            s_t=trajectory['state'][t]
            a_t=trajectory['action'][t]

            reward=trajectory['reward'][t]
            G=self.b_agent.gamma*G+reward
            
            self.C[s_t[0],s_t[1],a_t]+=W
            
            qv=self.t_agent.QValue[s_t[0],s_t[1],a_t]
            self.t_agent.QValue[s_t[0],s_t[1],a_t]=qv+(G-qv)*W/self.C[s_t[0],s_t[1],a_t]
            W=W*self.t_agent.Policy[s_t[0],s_t[1],a_t]/self.b_agent.Policy[s_t[0],s_t[1],a_t]
            
            if not W:
                break
    def run_n_episode(self,n_ep,start_op,dirpath):
        run_time=0        

        while(run_time != n_ep):            
            self.value_update(start_op)
            print(f'Episode:{run_time} Steps: {self.steps}')
            run_time+=1
            if run_time%200==0:
                print(' ############### Done',run_time,'Episodes  ###############')
           
                name=dirpath+'/PolicyUpdate_'+str(run_time)+'_'+time.strftime("%Y%m%d-%H%M%S")+'.png'
                self.t_agent.figPlotValue(name)

    def figOptPolicy(self,name): #update the policy at last for check
        non_opt_policy=np.copy(self.t_agent.Policy)
        for i in range(4):
            for j in range(4):
                self.t_agent.Policy[i,j]=self.t_agent.PolicyUpdate([i,j],'greedy')
        self.t_agent.figPlotMove(name)
        return non_opt_policy   
        

def main():    
    off_mc=Grid_MC_Off_Pre()

    start_op={'action':0,'state':[0,0]}

    dirpath='5/Off_Policy' #clean up the file dir
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    else:
        flist=[f for f in os.listdir(dirpath)]
        for f in flist:
            os.remove(os.path.join(dirpath,f))

    figname=dirpath+'/0'+'_MC_Off_Policy_Prediction.png' # intial policy
    off_mc.t_agent.figPlotMove(figname)

    off_mc.run_n_episode(1000,start_op,dirpath)

    
    figname1=dirpath+'/1'+'_MC_Off_Policy_Prediction.png' # change policy 
    off_mc.figOptPolicy(figname1)

if __name__=="__main__":
    main()