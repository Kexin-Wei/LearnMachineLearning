
import gym
import imageio
import os
import platform
import torch

from utils import Replay

def one_run(env_name,ac,len_sum,collect_len,args):
    env = gym.make(env_name)
    obs  = []
    acts = []
    obs_ = []
    rewards = []
    dones   = []
    advantages = []
    
    ob=env.reset()             
    step = 0
    while 1:
        act = ac.get_act(ob)
        ob_, reward, done, _ = env.step(act)

        obs.append(ob)
        obs_.append(ob_)
        acts.append(act)
        dones.append(done)
        rewards.append(reward)
        step += 1
        
        ob = ob_
        
        if step + len_sum >= collect_len or done:
            break
    
    advantages = ac.adv_cal(obs,obs_,dones,rewards,
                            args.GAMMA,args.LAMBDA)
    
    return obs, obs_, acts, dones, rewards, advantages

def agent(env_name,ac,args):
    memo = Replay()
    if ac.device == 'cpu':
        collect_len = args.STEPS_PER_EP / args.WORKER_NUM
    else:
        collect_len = args.STEPS_PER_EP
        
    if args.ONE_EP_FLAG:
        memo.append(*one_run(env_name,ac,-2e5,collect_len,args))
        return memo.return_all()
    
    while( memo.len < collect_len):
        memo.append(*one_run(env_name,ac,memo.len,collect_len,args))   
    return memo.return_all()

    
def test(ep,env_name,ac,RENDER_FLAG):
    env = gym.make(env_name)
    
    ob = env.reset()
    done = 0
    reward_sum = 0
    
    while not done:
        if RENDER_FLAG:
            env.render()
            
        act = ac.get_act(ob)
        ob_, reward, done, _ = env.step(act)
        ob = ob_
        reward_sum += reward
        
    env.close()
    return reward_sum

def train(memos,ac,ac_old,optimizer,args):
    kl_dict = {'max'  :[],
               'mean' :[]}
    
    loss_dict = {'clip'   :[],
                 'entropy' :[],
                 'vf'  :[],
                 'sum' :[]}
    
    for _ in range(args.UPDATE_EP):
        obs,obs_,acts,dones,rewards,advantages = memos.sample()        
        
        # loss 1: loss_clip
        dist_old,log_prob_old, _ = ac_old.get_policy(obs,acts)
        dist_new,log_prob_new, entropies_new = ac.get_policy(obs,acts)
        kl = torch.distributions.kl.kl_divergence(dist_new, dist_old)
        
       
        
        rs = torch.exp(log_prob_new-log_prob_old)
        
        advantages = torch.Tensor(advantages).to(ac.device)
        loss_clip = - torch.minimum(rs*advantages,
                        rs.clip(1-args.EPSILON,1+args.EPSILON)*advantages).mean()
        
        # loss 3: entropy
        loss_entropy = - args.COEF_ENTROPY * entropies_new.mean()
        
        # loss 2: loss_vf
        td_errors = ac.td_cal(obs,obs_,dones,rewards,args.GAMMA)
        loss_vf = args.COEF_VAL * (td_errors**2).mean()
        
        optimizer.zero_grad()
        (loss_clip+loss_vf+loss_entropy).backward()
        optimizer.step()
        
        kl_dict["max"].append(kl.max().item())
        kl_dict["mean"].append(kl.mean().item())
        loss_dict['clip'].append(loss_clip.item())
        loss_dict['entropy'].append(loss_entropy.item())
        loss_dict['vf'].append(loss_vf.item())
        loss_dict['sum'].append((loss_clip+loss_vf+loss_entropy).item())
        
    return kl_dict,loss_dict
    #ac_old.load_state_dict(ac.state_dict())
    
def dir_maker(args,FILENAME):
    comment = f"lr_{args.LR}"
    if args.ONE_EP_FLAG:
        comment = "one_ep_" + comment
    if not args.NORMALIZE_FLAG:
        comment += "_no_norm"
        
    comment += f"_ep_{args.EPOCHS}_update_{args.UPDATE_EP}_steps_{args.STEPS_PER_EP}"
    for i in range(len(args.fc_n)):        
        comment += f"_fc{i+1}_{args.fc_n[i]}"
    
    
    OS = "mac" if platform.system() == "Darwin" else "linux"
    DIR = os.path.join(f"test_{OS}_{FILENAME}_{args.env_name}",
                       comment)
    
    try:
        os.makedirs(DIR)
    except:
        print(f"Failed to open folder {DIR}")    
        
    return DIR,comment