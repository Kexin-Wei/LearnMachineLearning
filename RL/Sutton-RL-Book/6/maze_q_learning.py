# 2020.10.22
# q-learning
import numpy as np
import gym
import gym_maze
from agent import agent
import matplotlib.pyplot as plt
import os

def get_action(action_set,probability_set):
    return np.random.choice(action_set,p=probability_set)

def ob_maze(ob,maze_dim):
    return int(ob[0]*maze_dim[1]+ob[1])

def find_the_one(qvalue_set,act_dim,round_decimal,round_swtich):
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
    # return a ob_dim epsilon greedy probability                    
    p=np.ones(act_dim)*epsilon/act_dim
    theone=find_the_one(qvalue_set,act_dim,round_decimal,round_swtich)
    p[theone]+=1-epsilon
    return p

def argmax_a(qvalue_set,act_dim,round_decimal=3,round_swtich=True): 
    #return the one
    theone=find_the_one(qvalue_set,act_dim,round_decimal,round_swtich) 
    p=np.zeros(act_dim)
    p[theone]=1
    return p,theone

def makedir(dirpath): 
    if not os.path.exists(dirpath): #clean up the file dir
        os.makedirs(dirpath)
    else:
        flist=[f for f in os.listdir(dirpath)]
        for f in flist:
            os.remove(os.path.join(dirpath,f))
    return dirpath



def run():
    env=gym.make("maze-random-10x10-v0")

    #changable parameters
    #####################
    max_episode = 100
    alpha       = 0.5
    gamma       = 0.9
    epsilon     = 0.4
    #####################

    act_dim  = len(env.ACTION)
    maze_dim = env.maze_size
    ob_dim   = maze_dim[0]*maze_dim[1]

    # initial
    # policy: epsilon greedy
    a_agent  = agent(ob_dim,act_dim,gamma=gamma,epsilon=epsilon,\
                    policy='epsilon-soft',qvalue_range=3)
    terminal = ob_maze([maze_dim[0]-1,maze_dim[1]-1],maze_dim)
    a_agent.QValue[terminal,:] = 0

    ob = env.reset()
    env.render()
    input("start?")
    # run episodes
    for ep in range(max_episode):
        # episode_initial for agent
        ob = env.reset()
        ob_m = ob_maze(ob,maze_dim)
        endstep=0

        # run til terminal
        while(1): 
            env.render()            

            a_agent.Policy[ob_m] = epsilon_greedy(a_agent.QValue[ob_m],\
                                            a_agent.epsilon,act_dim,round_swtich=False)
            act=get_action(range(act_dim),a_agent.Policy[ob_m])

            ob,reward,done,_ = env.step(env.ACTION[act])
            ob_m_next = ob_maze(ob,maze_dim) # ob_m = next ob_m

            max_a=find_the_one(a_agent.QValue[ob_m_next],act_dim,3,False)            
            a_agent.QValue[ob_m,act] += alpha*(reward+\
                                        a_agent.gamma*a_agent.QValue[ob_m_next,max_a] \
                                        -a_agent.QValue[ob_m,act])            
            ob_m = ob_m_next

            endstep += 1        
            if done:
                break
            
                  
        print(f'Episode: {ep} Endstep: {endstep}')
        
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
    
