import gym
import gym_maze

def maze_action_input(ACTION):
    input_s=input('Choose:')
    action=ACTION[int(input_s)]
    return action


def human_test():
    env = gym.make("maze-random-5x5-v0")
    env.reset()
    env.render()
    
    act_dim=len(env.ACTION)
    maze_dim=env.maze_size
    max_episode=10
    max_steps=1000 # max steps in one episode

    print('Choose: 0.Up 1.Down 2.Right 3.Left')

    for t in range(max_steps):
        env.render()
        print('Step:',t,end=' ')
        action=maze_action_input(env.ACTION)
        observation,reward,done,info=env.step(action)
        print(observation,reward,done,info)
        if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

if __name__=="__main__":
    human_test()