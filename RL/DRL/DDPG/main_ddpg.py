import os
import gym
import numpy as np
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from noise import OUNoise
from model import Actor,Critic,DDPG_AC
from utils import get_args,dir_maker
from replay import ReplayOneDeque
from ddpg_agent import one_trajectory,train,eval_trajectory

def main():
    args = get_args()
    
    # directory define
    FILENAME = os.path.splitext(os.path.basename(__file__))[0]
    DIR,comment = dir_maker(args,FILENAME)
    BEST_MODEL_FILE = f"{DIR}/Best_Model.pt"
    LAST_MODEL_FILE = f"{DIR}/Last_Model.pt"
    writer = SummaryWriter(f"{DIR}/{comment}")    
    logging.basicConfig(filename=f'{DIR}/{comment}.log',level=logging.DEBUG)
    
    # Model define
    if args.DEVICE is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.DEVICE        
    env = gym.make(args.env_name)
    N_OB = env.observation_space.shape[0]
    N_ACT = env.action_space.shape[0]          
    ac = DDPG_AC(N_OB,N_ACT,fc_n=args.fc_n,device=device).to(device)
    ac_target = DDPG_AC(N_OB,N_ACT,fc_n=args.fc_n,device=device).to(device)
    ac_target.load_state_dict(ac.state_dict())        
    actor_optim = torch.optim.Adam(ac.actor.parameters(),lr= args.lr_actor)
    critic_optim = torch.optim.Adam(ac.critic.parameters(),lr= args.lr_actor)
    
    ou = OUNoise(mu=np.zeros(N_ACT))
    
    memos = ReplayOneDeque(MEMORY_SIZE=args.MEMORY_SIZE,BATCH_SIZE = args.BATCH_SIZE)
    
    # visulize data
    reward_list,eval_reward_list = [],[]
    loss_dict = {"critic":[],"actor":[]}
    best_reward = None
    
    # start train
    pbar = tqdm(range(args.EPOCHS))    
    for ep in pbar:
        
        reward_sum = one_trajectory(env,ac,ou,memos)
        
        writer.add_scalar("Reward",reward_sum,ep)
        reward_list.append(reward_sum)
        pbar_string = (f"Epoch: {ep} "
                       f"\t reward:{reward_sum:.2f}")
        
        loss_actor, loss_critic = train(memos,ac,ac_target,actor_optim,critic_optim,args)
            #breakpoint()         
        if loss_actor is not None:      
            loss_dict["critic"].append(loss_critic)
            loss_dict["actor"].append(loss_actor)
            writer.add_scalar("Loss_critic",loss_dict["critic"][-1],ep)
            writer.add_scalar("Loss_actor",loss_dict["actor"][-1],ep)
           
            pbar_string += (f"\tloss actor:{loss_actor:.5f}"
                            f"\tloss critic:{loss_critic:.5f}")
        
        if ep>0  and ep % args.EVAL_FREQ == 0:       
            eval_reward_sum = eval_trajectory(env,ac)
        
            eval_reward_list.append(eval_reward_sum)
            writer.add_scalar("Eval_reward",eval_reward_sum,ep)
            
            pbar_string += (f"\teval reward: {eval_reward_sum:.2f}"
                            f"\tmean reward: {sum(eval_reward_list[-100:])/min(len(eval_reward_list),100):.2f}"
                            f"\tbest reward: {max(eval_reward_list):.2f}")      
                  
            if best_reward is None or max(reward_list) > best_reward:
                best_reward = max(reward_list)  
 
        pbar.write(pbar_string)
        logging.info(pbar_string)
        writer.flush()
    
    torch.save(ac.state_dict(),LAST_MODEL_FILE)
    writer.close()
    
    plt.figure()
    plt.plot(reward_list)
    plt.title("Reward")
    plt.savefig(f"{DIR}/reward_list_and_max_{max(reward_list):.2f}.png")
    
    plt.figure()
    plt.plot(loss_dict['critic'],label= 'critic')
    plt.plot(loss_dict['actor'],label='actor')
    plt.legend(loc=2)
    plt.title("Loss")
    plt.savefig(f"{DIR}/loss_list_and_sum.png")

    
    print("Test Final Model:")
    reward_list = 0
    for i in tqdm(range(10)):
        reward_sum = eval_trajectory(env,ac)
        
        tqdm.write(f"\t {i} with reward: {reward_sum:.2f}")
        logging.info(f"\t {i} with reward: {reward_sum:.2f}")
        reward_list +=reward_sum
    print(f"Final Model Mean Reward:{reward_list/10:.2f}")
    logging.info(f"Final Model Mean Reward:{reward_list/10:.2f}")

    
if __name__ == "__main__":
    main()