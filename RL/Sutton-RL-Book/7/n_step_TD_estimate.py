# n-step TD
# 2020.11.29
import numpy as np
import gym
import gym_maze
from agent import agent
import matplotlib.pyplot as plt
import os

def get_action(action_set,probability_set):
    # input : a list type probability for 1 state
    # return: action chosen according to the probability
    return np.random.choice(action_set,p=probability_set)

def ob_maze(ob,maze_dim):
    # input : [i,j] 
    # return: corresponding state index
    return int(ob[0]*maze_dim[1]+ob[1])

def find_the_one(qvalue_set,act_dim,round_decimal,round_swtich):
    # input : a list type of qvalue at a state, round up decimal, 
    #         round_swtich to turn on / off the round up function
    # return: return one of the actions (number) having the biggest value
    if round_swtich:
        candidate=[]
        max_qv=round(max(qvalue_set),round_decimal) 
        #find the best +-0.009
        for i in range(act_dim):
            if round(qvalue_set[i],round_decimal) == max_qv:
                candidate.append(i)
    else:
        candidate=np.where(qvalue_set==max(qvalue_set))[0]
    return np.random.choice(candidate)

def epsilon_greedy(qvalue_set,epsilon,act_dim,\
                    round_decimal=3,round_swtich=True):
    # return: a epsilon greedy probability of one state
    p=np.ones(act_dim)*epsilon/act_dim
    theone=find_the_one(qvalue_set,act_dim,round_decimal,round_swtich)
    p[theone]+=1-epsilon
    return p

def argmax_a(qvalue_set,act_dim,round_decimal=3,round_swtich=True): 
    #return: the new greedy policy of one state, and the argmax_a action
    theone=find_the_one(qvalue_set,act_dim,round_decimal,round_swtich) 
    p=np.zeros(act_dim)
    p[theone]=1
    return p,theone

def makedir(dirpath): 
    # make dir for figures saving
    if not os.path.exists(dirpath): #clean up the file dir
        os.makedirs(dirpath)
    else:
        flist=[f for f in os.listdir(dirpath)]
        for f in flist:
            os.remove(os.path.join(dirpath,f))
    return dirpath

def store_rsa(store_dic,time,store_item,store_value):
    if time in store_dic:                   # old dictionary item with key time
        store_temple=store_dic[time]
    else:                                   # new dictionary item with key time
        store_temple={'r':0,'s':0,'a':0}
    store_temple[store_item]=store_value
    store_dic[time]=store_temple
    return store_dic

def run():
    env=gym.make("maze-random-10x10-v0")

    
    #####################
    #changable parameters
    max_episode = 100
    alpha       = 0.5   # step-size
    gamma       = 0.9   # G=R+gamma*G
    epsilon     = 0.4   # for policy setting
    n           = 20    # TODO:
    

    act_dim  = len(env.ACTION)
    maze_dim = env.maze_size
    ob_dim   = maze_dim[0]*maze_dim[1]

    #################
    # initial
    
    a_agent  = agent(ob_dim,act_dim,gamma=gamma,epsilon=epsilon,\
                    policy='random',qvalue_range=3)
    a_agent.PolicyUpdate(update='greedy',round_swtich=False)

    #################
    # run episodes
    for ep in range(max_episode):
        # episode_initial for agent
        ob   = env.reset()
        ob_m = ob_maze(ob,maze_dim)
        act  = get_action(range(act_dim),a_agent.Policy[ob_m])
        
        t = 0
        T = float('inf')
        
        store={0:{'r':0,'s':ob_m,'a':act}}
        #################
        # run til terminal
        while(1):            
            if t < T:
                env.render()
                ob,reward,done,_ = env.step(env.ACTION[act])
                ob_m_next = ob_maze(ob,maze_dim)            # ob_m = next ob_m
                store = store_rsa(store,t+1,'r',reward)     # store Rt+1 and St+1
                store = store_rsa(store,t+1,'s',ob_m_next)
                
                if done:
                    T = t+1
                else:
                    act   = get_action(range(act_dim),a_agent.Policy[ob_m_next])
                    store = store_rsa(store,t+1,'a',act)

            tau = t-n+1 # tau is the time whose estimate is being updated

            if tau >= 0:
            
                sum_list=np.arange(tau+1,min(tau+n,T))
                G=0
                for i in sum_list:
                    G+=a_agent.gamma**(i-tau-1)*store[i]['r']
                
                if tau + n < T:
                    G += a_agent.gamma**n\
                         *a_agent.QValue[store[tau+n]['s'],store[tau+n]['a']]

                qv = a_agent.QValue[store[tau]['s'],store[tau]['a']] 
                a_agent.QValue[store[tau]['s'],store[tau]['a']] += alpha*(G-qv)

                if pi is being learned:#TODO:
                    a_agent.PolicyUpdate(update='epsilon-greedy',round_swtich=False)
            
            t+=1
            if tau == T-1:
                break
                   
        print(f'Episode: {ep} Endstep: {t}')
    
    #################    
    # find a better way to know the results
    a_agent.PolicyUpdate(round_swtich=False)

    ob=env.reset()
    ob_m=ob_maze(ob,maze_dim)
    for t in range(1000):
        env.render()
        action=get_action(range(act_dim),a_agent.Policy[ob_m])
        observation,reward,done,info=env.step(env.ACTION[action])
        ob_m=ob_maze(observation,maze_dim)
        if done:
            break

if __name__=="__main__":
    run()
    