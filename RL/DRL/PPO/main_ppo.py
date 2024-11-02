
import os
import gym
import concurrent.futures
import time

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from model import ActorCritic
from utils import Replay,get_args
from functions import agent,test,dir_maker,train

def main():
    args = get_args()
    
    # directory define
    FILENAME = os.path.splitext(os.path.basename(__file__))[0]
    DIR,comment = dir_maker(args,FILENAME)
    MODEL_FILE = f"{DIR}/Best_Model.pt"
    

    # network define            
    if args.DEVICE is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'        
    else:
        device = args.DEVICE
    
    env = gym.make(args.env_name)
    N_OB = env.observation_space.shape[0]
    N_ACT = env.action_space.shape[0]          
        
    ACNet = ActorCritic(N_OB,N_ACT,fc_n=args.fc_n,device=device,NORMALIZE = args.NORMALIZE_FLAG)
    if os.path.isfile(MODEL_FILE):
        ACNet.load_state_dict(torch.load(MODEL_FILE))
        print("Find old model, use the old as start")
        
    ACNet_old = ActorCritic(N_OB,N_ACT,fc_n=args.fc_n,device=device,NORMALIZE = args.NORMALIZE_FLAG)
    ACNet_old.load_state_dict(ACNet.state_dict())
    optimizer = torch.optim.Adam(ACNet.parameters(),lr=args.LR)
        
    
    # visulize data
    reward_list = []
    best_reward = None
    kl_max = []
    loss_sum = { 'sum': [],
                  'vf': [],
                  'entropy': []}
        
    
    if args.TENSORBOARD_FLAG:
        writer = SummaryWriter(f"{DIR}/{comment}")
    
    # start train
    pbar = tqdm(range(args.EPOCHS))
    for ep in pbar:
    
        memos = Replay(BATCH_SIZE = args.BATCH_SIZE)
        # past = time.perf_counter()
        # obs, acts, td_errors, advantages = agent(args.env_name,ACNet,args)
        if torch.cuda.is_available():
            memos.append(*agent(args.env_name,ACNet,args))
        else:                
            executor=concurrent.futures.ThreadPoolExecutor(max_workers = args.WORKER_NUM )
            results = executor.map(agent,[args.env_name]*args.WORKER_NUM,
                                            [ACNet]*args.WORKER_NUM,
                                            [args]*args.WORKER_NUM)
            for result in results:
                memos.append(*result)
            del executor
            
            # agents = [executor.submit(agent,args.env_name,ACNet,args) for _ in range(args.WORKER_NUM)]                
            # for f in concurrent.futures.as_completed(agents):
            #     if ep ==0: memos.append(*f.result())
            #     print(len(f.result()))
            # executor.join()
            #executor.shutdown(True) 
        # now = time.perf_counter()
        # print(f"Second {now - past:.2f}, Len {memos.len}")
        
        kl_dict,loss_dict = train(memos,ACNet,ACNet_old,optimizer,args)
        kl_max.append(sum(kl_dict["max"])/args.UPDATE_EP)
        loss_sum['vf'].append(sum(loss_dict['vf']))
        loss_sum['sum'].append(sum(loss_dict['sum']))
        loss_sum['entropy'].append(sum(loss_dict['entropy']))
        
        ACNet_old.load_state_dict(ACNet.state_dict())
        
        if ep>0 and ep % args.EVAL_FREQUENCY == 0 :
            reward_sum = test(ep,args.env_name,ACNet,args.RENDER_FLAG)
            reward_list.append(reward_sum)
            
            
            pbar_string= f"Epoch: {ep}" + \
                         f"\treward: {reward_sum:.2f}" + \
                         f"\tmean reward: {sum(reward_list[-100:])/min(len(reward_list),100):.2f}"  + \
                         f"\tbest reward: {max(reward_list):.2f}"
            
            if best_reward is None or max(reward_list) > best_reward:
                best_reward = max(reward_list)
                pbar_string += "\t model saved"
                torch.save(ACNet.state_dict(),MODEL_FILE)                

            if args.TENSORBOARD_FLAG:   
                writer.add_scalar("Reward",reward_sum,ep)
                writer.flush()
            pbar.write(pbar_string)
            
            
        if args.TENSORBOARD_FLAG:
            writer.add_scalar("KL_max",kl_max[-1],ep)
            writer.add_scalar("Loss_sum_sum",loss_sum['sum'][-1],ep)
            writer.add_scalar("Loss_vf_sum",loss_sum['vf'][-1],ep)
            writer.add_scalar("Loss_entropy_sum",loss_sum['entropy'][-1],ep)
            writer.flush()
            
        pbar.set_postfix_str(s=f" kl:{kl_max[-1]:.3e},  loss:{loss_sum['sum'][-1]:.3f}")
            
    
    
    if args.TENSORBOARD_FLAG:
        writer.close()
    
    plt.figure()
    plt.plot(reward_list)
    plt.title("Reward")
    plt.savefig(f"{DIR}/reward_list_and_max_{max(reward_list):.2f}.png")
    
    plt.figure()
    plt.plot(kl_max)
    plt.title("KL Divergence")
    plt.savefig(f"{DIR}/kl_list_and_max_{max(kl_max):.2f}.png")
    
    plt.figure()
    plt.plot(loss_sum['vf'],label= 'vf')
    plt.plot(loss_sum['sum'],label='sum')
    plt.plot(loss_sum['entropy'],label='entropy')
    plt.legend(loc=2)
    plt.title("Loss")
    plt.savefig(f"{DIR}/loss_list_and_sum.png")

    
    print("Test Final Model:")
    reward_sum = 0
    for i in tqdm(range(10)):
        reward = test('Final',args.env_name,ACNet,args.RENDER_FLAG)
        tqdm.write(f"\t {i} with reward: {reward:.2f}")
        reward_sum += reward
    print(f"Final Model Mean Reward:{reward_sum/10:.2f}")

        
    ACNet.load_state_dict(torch.load(MODEL_FILE))
    print("Test Best Model:")
    reward_sum = 0
    for i in tqdm(range(10)):
        reward = test('Final',args.env_name,ACNet,args.RENDER_FLAG)
        tqdm.write(f"\t {i} with reward: {reward:.2f}")
        reward_sum += reward
    print(f"Best Model Mean Reward:{reward_sum/10:.2f}")
    
if __name__ == "__main__":
    main()