import gym
import numpy as np
import matplotlib.pyplot as plt
from agent_util import *

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
    #=========== Edit ========================
    GAMMA   = 0.9
    ALPHA   = 0.8
    EPSILON = 0.5
    N_EP    = 10000
    #=========== Initial ======================
    env = gym.make(env_name)

    N_ACT = env.action_space.n
    N_OB  = env.observation_space.n

    Qvalue=np.zeros([N_OB,N_ACT])
    
    list_reward = []
    #=========== Train ======================
    for i in range(N_EP):
        ob = env.reset()
        sum_reward = 0

        while(1):
            env.render()

            act=chose_action(Qvalue[ob],N_ACT,EPSILON)

            ob_next, reward, done, info = env.step(act)

            Qvalue[ob,act] += ALPHA *(reward +  \
                                      GAMMA* max(Qvalue[ob_next])\
                                      - Qvalue[ob,act])
            
            ob = ob_next

            sum_reward += reward            

            if done:
                break
        list_reward.append(sum_reward)

    plt.plot(list_reward)
    plt.show()
    print(Qvalue)

    # take the greedy
    ob = env.reset()
    while(1):
        env.render()

        act=chose_action(Qvalue[ob],N_ACT,0)

        ob_next, reward, done, info = env.step(act)

        ob = ob_next

        if done:
            break

    env.close()

if __name__=="__main__":
    env_name="FrozenLake-v0"
    train(env_name)