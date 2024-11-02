from Grid_env import Grid
import numpy as np
import os
import time
"""
algorithm:
    run n episode:
        according policy get trajectory[s0,a0,r1,s1,a1,r2,.....rT]
        calculate G (backwards for loop): t:T-1->0
            G=gamma*G+rt
            Q_n_episode.append(G)
            Q=average(Q_n_episode)
            policy update (greedy or epsilon-soft)
"""

class Grid_MC_OP(Grid):
    def __init__(self, gamma=0.9,Terminal=[[0,3]],
                       policy='ep_soft',epsilon=0.3,**kwargs): # policy options see Grid_env.py
        super().__init__(gamma,Terminal,policy)        
        self.Return_n=np.zeros((4,4,4)) # index: state_0,state_1,action
        self.steps=0
        self.epsilon=epsilon
        self.run_time=0
        if policy=='ep_soft':
            self.update_op='ep_soft'
        else:
            self.update_op='greedy' #only choose the optimal using Action=argmax_a Q(s,a)

    
        
    def is_s_a_in_trajectory(self,trajectory,state,action,t):
        list_a=trajectory['action'][:(t-1)]
        list_s=trajectory['state'][:(t-1)]

        if action in list_a and  state in list_s: # if they are all in, check if there is a pair
            a_index = [i for i, a in enumerate(reversed(list_a)) if a == action] # from t-1->0
            for j in a_index:
                if list_s[j]==state: # pair matchs
                    return True
        return False

    def explore_start(self): # no use for this func
        while(1):
            s_0=np.random.randint(4,size=2).tolist()
            a_0=np.random.randint(4,size=1)[0]
            p=self.Policy[s_0[0],s_0[1],a_0]
            if p > 0:
                return s_0,a_0


    def run_one_episode(self,start_state=[0,0],start_action=0):
        trajectory={'state':[],
                    'action':[],
                    'reward':[]}

        state=start_state
        action=start_action

        self.steps=0
        while(1):                                               
            next_s=self.NextState(action,state)
            reward=self.getReward(state,next_s)

            trajectory['state'].append(state)
            trajectory['action'].append(action) 
            trajectory['reward'].append(reward)

            if self.IsTerminal(next_s):                
                break
            state=next_s
            action=self.Action(state)
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
        for i in range(1,self.steps): # i=0,1,2,...T-1
            t=-i
            reward=trajectory['reward'][t]
            G=self.gamma*G+reward
            s_t=trajectory['state'][t]
            a_t=trajectory['action'][t]
            if not self.is_s_a_in_trajectory(trajectory,s_t,a_t,t):
                self.Return_n[s_t[0],s_t[1],a_t]+=1
                v=self.QValue[s_t[0],s_t[1],a_t]                
                self.QValue[s_t[0],s_t[1],a_t]=v+(G-v)/self.Return_n[s_t[0],s_t[1],a_t]
                #print(self.QValue[s_t[0],s_t[1],a_t])
                self.Policy[s_t[0],s_t[1]]=self.PolicyUpdate(s_t,self.update_op)
    
    def run_n_episode(self,n_ep,start_op,dirpath):
        self.run_time=0        
        while(self.run_time != n_ep):  

            self.value_update(start_op)
            print(f'Gamma: {self.gamma} Ep: {self.epsilon} Episode:{self.run_time} Steps: {self.steps}')
            
            self.run_time+=1
            if self.run_time%500==0:
                print('############### Done',self.run_time,'Episodes  ###############')
                name=dirpath+'/PolicyUpdate_'+str(self.run_time)+'_'+time.strftime("%Y%m%d-%H%M%S")+'.png'
                self.figPlotValue(name)
                
        
    def figOptPolicy(self,name): #update the policy at last for check
        non_opt_policy=np.copy(self.Policy)
        for i in range(4):
            for j in range(4):
                self.Policy[i,j]=self.PolicyUpdate([i,j],'greedy')
        self.figPlotMove(name)
        return non_opt_policy
        

def main():    
    gamma=[1,0.9] # not change that much, but 0.9 is well than 0.7
    epsilon=[1,0.8,0.7] # not explore enougth when ep<0.4
    g=0.9

    dirpath='5/OP' #clean up the file dir
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    else:
        flist=[f for f in os.listdir(dirpath)]
        for f in flist:
            os.remove(os.path.join(dirpath,f))

    for g in gamma:
        for e in epsilon:
            agent=Grid_MC_OP(gamma=g,epsilon=e,policy='ep_soft',Terminal=[[3,0]])
            
            figname=dirpath+'/0_MC_OP_gamma_'+str(g)+'_ep_'+str(e)+'.png'
            agent.figPlotValue(figname)

            start_op={'action':0,'state':[0,0]}
            agent.run_n_episode(2000,start_op,dirpath)

            t=agent.run_time
            #save the unchanged Qvalue 
            figname=dirpath+'/1_MC_OP_gamma_'+str(g)+'_ep_'+str(e)+'_run_time_'+str(t)+'.png'
            agent.figPlotValue(figname)
            #save the opt policy according to the last time QValue
            figname=dirpath+'/2_MC_OP_gamma_'+str(g)+'_ep_'+str(e)+'_run_time_'+str(t)+'.png'            
            non_opt_policy=agent.figOptPolicy(figname)

if __name__=="__main__":
    main()