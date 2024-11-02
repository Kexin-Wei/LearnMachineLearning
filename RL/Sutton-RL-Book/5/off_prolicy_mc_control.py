from Grid_env import Grid
import numpy as np
import os
import time

class Grid_MC_Off_Control():
    def __init__(self, gamma=0.9,Terminal=[[0,3]],epsilon=0.5,b_mode='ep_soft',**kwargs): 
                       # policy options see Grid_env.py
        self.b_agent=Grid(gamma,Terminal)        
        
        t_policy,QValue=self.initial_targert_policy() # b_agent has the same QValue with t_agent
        self.t_agent=Grid(gamma,Terminal,policy=t_policy,qvalue=QValue)  # use b_agent QValue to generate t_agent policy
        
        self.C=np.zeros((4,4,4)) # index: state_0,state_1,action
        self.steps=0
        self.epsilon=epsilon
        self.b_mode=b_mode
        


    def initial_targert_policy(self): # greedy update the target policy
        t_policy=np.zeros((4,4,4))

        QValue=np.copy(self.b_agent.QValue)

        for i in range(4):
            for j in range(4):
                max_2 = round(max(QValue[i,j]), 2)
                candidate = [] # randomly choose a max one *accuracy:+-0.01
        
                for a in range(4):
                    if round(QValue[i,j,a], 2) == max_2: candidate.append(a)
                theone=np.random.choice(candidate) # the argmax_a
        
                p=np.zeros(4)
                p[theone]=1
        
                t_policy[i,j]=p
        return t_policy,QValue




    def initial_behaviour_policy(self):        
        if self.b_mode=='best':
            for i in range(4):
                for j in range(4):
                    non_zero_index=np.nonzero(self.t_agent.Policy[i,j])
                    off_set=[ 1 if k in non_zero_index[0] else 0 for k in range(4)]
                    p=np.random.rand(4)+np.array(off_set)
                    p=p/sum(p)
                    self.b_agent.Policy[i,j]=p
        else:
            for i in range(4):
                for j in range(4):
                    theone=np.random.choice(4) # with random soft policy
                    p=np.ones(4)*self.epsilon/4
                    p[theone]+=1-self.epsilon
                    self.b_agent.Policy[i,j]=p
        


    def explore_start(self): # no use for this func
        while(1):
            s_0=np.random.randint(4,size=2).tolist()
            a_0=np.random.randint(4,size=1)[0]
            p=self.b_agent.Policy[s_0[0],s_0[1],a_0]
            if p > 0:
                return s_0,a_0




    def run_one_episode(self,start_state=[0,0],start_action=0):

        self.initial_behaviour_policy()

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

            G=self.b_agent.gamma*G+reward #G=gamma*G+R
            
            
            self.C[s_t[0],s_t[1],a_t]+=W # C=C+W
            
            qv=self.t_agent.QValue[s_t[0],s_t[1],a_t]
            self.t_agent.QValue[s_t[0],s_t[1],a_t]=qv+ \
                                (G-qv)*W/self.C[s_t[0],s_t[1],a_t]
            self.t_agent.Policy[s_t[0],s_t[1]]=self.t_agent.PolicyUpdate(s_t)

            max_choice=np.random.choice(self.t_agent.Policy[s_t[0],s_t[1]])
            if max_choice!=a_t:
                break
            W=W/self.b_agent.Policy[s_t[0],s_t[1],a_t]

            





    def run_n_episode(self,n_ep,start_op,dirpath):
        run_time=0        

        while(run_time != n_ep):      

            self.value_update(start_op)


            print(f'Gamma: {self.t_agent.gamma} Ep: {self.epsilon} Episode:{run_time} Steps: {self.steps}')
            run_time+=1      

            if run_time%2000==0:
                print('############### Done',run_time,'Episodes  ###############') 

            name=dirpath+'/t_agent_QValue_gamma_'+str(self.t_agent.gamma)+ \
                        '_ep_'+str(self.epsilon)+'_'+\
                        str(run_time)+'_'+time.strftime("%Y%m%d-%H%M%S")+'.png'
            self.t_agent.figPlotValue(name)



    def figOptPolicy(self,name): #update the policy at last for check
        non_opt_policy=np.copy(self.t_agent.Policy)
        for i in range(4):
            for j in range(4):
                self.t_agent.Policy[i,j]=self.t_agent.PolicyUpdate([i,j],'greedy')
        self.t_agent.figPlotMove(name)
        return non_opt_policy   
        

        

if __name__=="__main__":
    gamma=[1,0.9] # not change that much, but 0.9 is well than 0.7
    epsilon=[1,0.8,0.7] # not explore enougth when ep<0.4

    
    dirpath='5/Off_Policy_Ctrl' #clean up the file dir
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    else:
        flist=[f for f in os.listdir(dirpath)]
        for f in flist:
            os.remove(os.path.join(dirpath,f))                    
    
    
    for g in gamma:
        for e in epsilon:
            
            off_mc=Grid_MC_Off_Control(epsilon=e,gamma=g,Terminal=[[3,3]])
            start_op={'action':0,'state':[0,0]}

            figname=dirpath+'/0'+'_MC_OP_C_Move_gamma_'+str(g)+'_ep_'+str(e)+'.png' # intial policy
            off_mc.t_agent.figPlotMove(figname,)
            figname=dirpath+'/0'+'_MC_OP_C_QValue_gamma_'+str(g)+'_ep_'+str(e)+'.png' # intial QValue
            off_mc.t_agent.figPlotValue(figname)

            off_mc.run_n_episode(8000,start_op,dirpath)
            
            figname=dirpath+'/1'+'_MC_OP_C_Move_gamma_'+str(g)+'_ep_'+str(e)+'.png' # changed policy 
            off_mc.t_agent.figPlotMove(figname)
            figname=dirpath+'/1'+'_MC_OP_C_QValue_gamma_'+str(g)+'_ep_'+str(e)+'.png' # changed QValue 
            off_mc.t_agent.figPlotValue(figname)

            