from Grid_env import Grid
import numpy as np

class Grid_MC_ES(Grid):
    def __init__(self, gamma=0.9,Terminal=[[0,3]],**kwargs):
        super().__init__(gamma=gamma,Terminal=Terminal)
        self.Return_n=np.zeros((4,4,4)) # index: state_0,state_1,action
        self.steps=0

    def argmax(self,state,action):        
        # find the max, when the rest is smaller, put it to 0
        Q_s_a=self.QValue[state[0],state[1]]
        max_2 = round(max(Q_s_a), 2)
        a_range = len(Q_s_a)
        candidate = [] # randomly choose a max one *accuracy:+-0.01
        for a in range(a_range):
            if round(Q_s_a[a], 2) == max_2: candidate.append(a)
        theone=np.random.choice(candidate)
        pi_new=np.zeros(4)
        pi_new[theone]=1
        return pi_new
    
    def IsSAinTrj(self,trajectory,state,action,t):
        list_a=trajectory['action'][:(t-1)]
        list_s=trajectory['state'][:(t-1)]

        if action in list_a and  state in list_s: # if they are all in, check if there is a pair
            a_index = [i for i, a in enumerate(reversed(list_a)) if a == action] # from t-1->0
            for j in a_index:
                if list_s[j]==state: # pair matchs
                    return True
        return False

    def ExploreStart(self):
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
            if self.steps > 2000:
                    if self.steps%100==0:
                        print('No ending!!! Now has run',self.steps,'steps')
                        if self.steps>1e4:
                            
                            break               

        print(f'End with {self.steps} Steps')    
        return trajectory
    
    def Value_Update(self):
        s_0,a_0=self.ExploreStart()
        trajectory=self.run_one_episode(start_action=a_0,start_state=s_0)

        G=0
        for i in range(1,self.steps): # i=1,2,...T-1
            t=-i-1
            reward=trajectory['reward'][t+1]
            G=self.gamma*G+reward
            s_t=trajectory['state'][t]
            a_t=trajectory['action'][t]
            if not self.IsSAinTrj(trajectory,s_t,a_t,t):
                self.Return_n[s_t[0],s_t[1],a_t]+=1
                v=self.QValue[s_t[0],s_t[1],a_t]
                self.QValue[s_t[0],s_t[1],a_t]=v+(G-v)/self.Return_n[s_t[0],s_t[1],a_t]
                self.Policy[s_t[0],s_t[1]]=self.argmax(s_t,a_t)
    
    def run_n_episode(self,n_ep):
        run_time=0
        while(run_time is not n_ep):
            print(f'############ Episode {run_time} ###############')
            self.Value_Update()
            run_time+=1

            if self.steps>1e4:
                print('This policy can\'t be improved anymore! and not optimal!')   
                break

            if run_time%20==0:
                print('Done',run_time,'Episodes..')

            name='5/ES/PolicyUpdate_'+str(run_time)+'.png'
            self.figPlotMove(name)
            
        

def main():
    
    policy=np.ones((4,4,4))*0.25
    agent=Grid_MC_ES(Policy=policy)
    agent.figPlotValue('5/ES/0_MC_ES.png',value='qv')
    agent.run_n_episode(100)
    agent.figPlotValue('5/ES/1_MC_ES.png',value='qv')


if __name__=="__main__":
    main()