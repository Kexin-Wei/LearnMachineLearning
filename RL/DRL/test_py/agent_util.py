import numpy as np

def PolicyUpdate(agent,update='greedy',update_depend='q',round_swtich=False,round_decimal=3):
        '''
        update options:
            1. greedy
            2. epsilon-greedy
            3. epsilon-soft
            4. set value
        update dependence:
            1. qvalue
            2. value
        '''
        type_ud=type(update)
        if type_ud==str:
            if update=='greedy':
                for i in range(agent.ob_dim):
                    agent.Policy[i]=np.zeros(agent.act_dim)
                    theone=find_the_one(agent.QValue[i],agent.act_dim,round_decimal,round_swtich)
                    agent.Policy[i,theone]=1
            if update=='epsilon-soft':
                for i in range(agent.ob_dim):
                    # p > epsilon/agent.act_dim, rest=1-epsilon
                    agent.Policy[i]=np.random.dirichlet(np.ones(agent.act_dim))\
                                    *(1-agent.epsilon)+agent.epsilon/agent.act_dim
            if update=='epsilon-greedy':
                for i in range(agent.ob_dim):
                    agent.Policy[i]=np.ones(agent.act_dim)*agent.epsilon/agent.act_dim
                    theone=find_the_one(agent.QValue[i],agent.act_dim,round_decimal,round_swtich)
                    agent.Policy[i,theone]+=1-agent.epsilon    

def find_the_one(qvalue_set,act_dim,\
                 round_decimal,round_swtich):
        '''
        help to find the candidate while policy update
        '''        
        if round_swtich:
            candidate=[]
            max_qv=round(max(qvalue_set),round_decimal) 
            #find the best +-0.009
            for i in range(act_dim):
                if round(qvalue_set[i],round_decimal) == max_qv:
                    candidate.append(i)
        else:
            candidate=np.where(qvalue_set==max(qvalue_set))[0]
        return np.random.choice(candidate) #return the one

class Agent:
    def __init__(self, ob_dim,act_dim,\
                       epsilon=0.3,\
                       policy='random',\
                       value='random',value_range=2,\
                       qvalue='random',qvalue_range=1):
        
        self.epsilon=epsilon
        self.ob_dim=ob_dim  # num of states
        self.act_dim=act_dim # num of actions
        
        self.ValueInitial(value,value_range)
        self.QValueInitial(qvalue,qvalue_range)
        self.PolicyInitial(policy)        

    def ValueInitial(self,value,value_range):
        '''
        value options:
            1. zero
            2. random: need value_range
            3. set value
        '''
        type_v=type(value)
        self.Value=np.zeros(self.ob_dim)
        if type_v==str:
            if value=='zero':
                pass
            if value=='random':
                self.Value=np.random.randint(value_range,size=self.ob_dim)\
                            -value_range/2 #[-qvalue_range/2,+qvalue_range/2]
        if type_v==np.ndarray:
            self.Value=value


    def QValueInitial(self,qvalue,qvalue_range):
        '''
        qvalue options:
            1. zero
            2. random: need qvalue_range
            3. set value
        '''
        type_qv=type(qvalue)
        self.QValue=np.zeros((self.ob_dim,self.act_dim))
        if type_qv==str:
            if qvalue=='zero':
                pass
            if qvalue=='random':
                self.QValue=np.random.randint(qvalue_range,size=(self.ob_dim,self.act_dim))\
                            -qvalue_range/2 #[-qvalue_range/2,+qvalue_range/2]
        if type_qv==np.ndarray:
            self.QValue=qvalue
        

    def PolicyInitial(self,policy):
        '''
        policy option:
            1. random
            2. epsilon-soft
            3. set value
        '''
        type_p=type(policy)
        self.Policy=np.zeros((self.ob_dim,self.act_dim))

        if type_p==str:
            if policy=='random':
                for i in range(self.ob_dim):
                    self.Policy[i]=np.random.dirichlet(np.ones(self.act_dim)) # random [0,1], sum==1

            if policy=='epsilon-soft':
                for i in range(self.ob_dim):
                    # p > epsilon/self.act_dim, rest=1-epsilon
                    self.Policy[i]=np.random.dirichlet(np.ones(self.act_dim))\
                                    *(1-self.epsilon)+self.epsilon/self.act_dim      
        if type_p==np.ndarray:
            self.Policy=policy


    







        