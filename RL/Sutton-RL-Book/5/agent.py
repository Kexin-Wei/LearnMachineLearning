import numpy as np

class agent:
    def __init__(self, ob_dim,act_dim,\
                        gamma=0.9,policy='random',\
                        epsilon=0.3,qvalue='random',qvalue_range=1):
        
        self.gamma=gamma
        self.epsilon=epsilon
        self.ob_dim=ob_dim  # num of states
        self.act_dim=act_dim # num of actions
        self.QValueInitial(qvalue,qvalue_range)
        self.PolicyInitial(policy)


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


    def find_the_one(self,qvalue_set,round_decimal=3):
        '''
        help to find the candidate while policy update
        '''
        candidate=[]

        max_2=round(max(qvalue_set),round_decimal) #find the best +-0.001
        for i in range(self.act_dim):
            if round(qvalue_set[i],round_decimal) == max_2:
                candidate.append(i)
        return np.random.choice(candidate) #return the one


    def PolicyUpdate(self,update='greedy',update_depend='q'):
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
                for i in range(self.ob_dim):
                    self.Policy[i]=np.zeros(self.act_dim)
                    theone=self.find_the_one(self.QValue[i])
                    self.Policy[i,theone]=1
            if update=='epsilon-soft':
                for i in range(self.ob_dim):
                    # p > epsilon/self.act_dim, rest=1-epsilon
                    self.Policy[i]=np.random.dirichlet(np.ones(self.act_dim))\
                                    *(1-self.epsilon)+self.epsilon/self.act_dim
            if update=='epsilon-greedy':
                for i in range(self.ob_dim):
                    self.Polciy[i]=np.ones(self.act_dim)*self.epsilon/self.act_dim
                    theone=self.find_the_one(self.QValue[i])
                    self.Policy[i,theone]+=1-self.epsilon







        