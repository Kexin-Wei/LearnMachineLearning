import numpy as np
import matplotlib.pyplot as plt
import gym


def chose_action(q_value_set,n_act,epsilon,round_switch=False,round_decimal=3):
    # choose the action with epsilon greedy
    theone=find_the_one(q_value_set,n_act,round_switch,round_decimal)
    p_qvalue=np.ones(n_act)*epsilon/n_act
    p_qvalue[theone]+=1-epsilon
    return np.random.choice(range(n_act),p=p_qvalue)

def find_the_one(qvalue_set,act_dim,\
                 round_switch,round_decimal):
        '''
        help to find the candidate while policy update
        '''        
        if round_switch:
            candidate=[]
            max_qv=round(max(qvalue_set),round_decimal) 
            #find the best +-0.009
            for i in range(act_dim):
                if round(qvalue_set[i],round_decimal) == max_qv:
                    candidate.append(i)
        else:
            candidate=np.where(qvalue_set==max(qvalue_set))[0]
        return np.random.choice(candidate) #return the one

def train(env_name):
    #======== Edit ============
    EPOCHS  = 10
    EPSILON = 0.3
    #==========================
    env = gym.make(env_name)
    
    N_ACT = env.action_space.n 
    N_OB  = env.observation_space.n 

    # 2 layers : input layer + output layer
    weights = np.random.rand(N_OB,N_ACT)
    bias    = np.zeros(N_ACT) 

    

    for i in range(EPOCHS):
        ob = env.reset()

        while(1):
            env.render()
            
            # build input 
            inputs = np.zeros(N_OB)
            inputs[ob] = 1 # [0 0 0 ... 0 1 0 ... 0 0]

            # predict
            q = np.dot(inputs,weights) + bias
            
            # action base epsilon-greedy
            act = chose_action(q,N_ACT,EPSILON)

            # get reward
            ob_next, reward, done, info = env.step(act)

            # update network

            if done:
                break
    
    env.close()

if __name__ == "__main__":
    env_name = "MountainCar-v0"
    train(env_name)