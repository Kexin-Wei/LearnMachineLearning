# 2020.10.21
# not that well to proof if right or not
# better try sarsa
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

def argmax_a(qvalue_set,act_dim,round_decimal=3,round_swtich=True): #no use
    if round_swtich:
        candidate=[]
        max_qv=round(max(qvalue_set),round_decimal) 
        #find the best +-0.009
        for i in range(act_dim):
            if round(qvalue_set[i],round_decimal) == max_qv:
                candidate.append(i)
    else:
        candidate=np.where(qvalue_set==max(qvalue_set))
    
    theone=np.random.choice(candidate) #return the one
    p=np.zeros(act_dim)
    p[theone]=1
    return p,theone

def makedir(dirpath): # no use
    if not os.path.exists(dirpath): #clean up the file dir
        os.makedirs(dirpath)
    else:
        flist=[f for f in os.listdir(dirpath)]
        for f in flist:
            os.remove(os.path.join(dirpath,f))
    return dirpath



def run():
    env=gym.make("maze-random-3x3-v0")

    #changable parameters
    #####################
    max_episode = 100
    alpha       = 0.5
    gamma       = 0.9
    #####################

    act_dim  = len(env.ACTION)
    maze_dim = env.maze_size
    ob_dim   = maze_dim[0]*maze_dim[1]

    # initial
    a_agent  = agent(ob_dim,act_dim,gamma=gamma,policy='epsilon-soft')
    terminal = ob_maze([maze_dim[0]-1,maze_dim[1]-1],maze_dim)
    a_agent.Value[terminal] = 0

    # run episodes
    for ep in range(max_episode):
        # episode_initial for agent
        ob = env.reset()
        ob_m = ob_maze(ob,maze_dim)
        endstep=0

        # run til terminal
        while(1): 
            env.render()
            act = get_action(env.ACTION,a_agent.Policy[ob_m])

            ob,reward,done,_ = env.step(act)

            ob_m_next = ob_maze(ob,maze_dim) # ob_m = next ob_m
            
            a_agent.Value[ob_m] = a_agent.Value[ob_m]+ alpha* \
                                    (reward+a_agent.gamma*a_agent.Value[ob_m_next] \
                                        -a_agent.Value[ob_m])
            
            ob_m=ob_m_next
            endstep+=1        
            if done:
                break
            
                  
        print(f'Episode: {ep} Endstep: {endstep}')
        
    # find a better way to know the results
    value_maze=np.zeros((maze_dim[0],maze_dim[1]))
    for i in range(ob_dim):
        dim_0=i//maze_dim[0]
        dim_1=i%maze_dim[1]
        value_maze[dim_0,dim_1]=a_agent.Value[i]
    plt.figure()
    plt.imshow(value_maze)
    plt.show()
    
    

if __name__=="__main__":
    run()
    