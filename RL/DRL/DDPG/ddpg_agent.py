import torch
import torch.nn.functional as F
def one_trajectory(env,ac,ou,memos):
    ou.reset()
    reward_sum = 0
    # collection the memos
    ob = env.reset()
    while 1:
        env.render()
        act = ac.get_act_numpy(ob)+ou()
        ob_,reward,done,_ = env.step(act)
        memos.append(ob,ob_,act,done,reward)
        ob = ob_
        reward_sum += reward
        if done:                
            return reward_sum
        
        
def train(memos,ac,ac_target,actor_optim,critic_optim,args):
    # train the network
    if memos.len>args.BATCH_SIZE:
        
        _,obs,obs_,acts,dones,rewards = memos.sample() 
        rewards = torch.Tensor(rewards).view(args.BATCH_SIZE,-1).to(ac.device)
        acts    = torch.Tensor(acts).to(ac.device)
        
        actor_optim.zero_grad()
        predict_acts = ac.get_act(obs)
        predict_qs = ac.get_qs(obs,predict_acts)
        loss_actor = - predict_qs.mean()            
        loss_actor.backward()
        actor_optim.step()
        
        #breakpoint()
        critic_optim.zero_grad()
        acts_ = ac_target.get_act(obs_)        
        target_qs = rewards + args.GAMMA * ac_target.get_qs(obs_,acts_)
        loss_critic = F.mse_loss(ac.get_qs(obs,acts),target_qs)           
        loss_critic.backward()
        critic_optim.step()                        
        
        #breakpoint()
        for target_param, param in zip(ac_target.actor.parameters(), ac.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - args.TAU) + param.data * args.TAU)
        for target_param, param in zip(ac_target.critic.parameters(), ac.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - args.TAU) + param.data * args.TAU)

        return loss_actor.item(), loss_critic.item()
    else:
        return None, None
    
def eval_trajectory(env,ac):      
    ob = env.reset()
    eval_reward_sum = 0
    while 1:
        env.render()
        act = ac.get_act_numpy(ob)
        ob_,reward,done,_ = env.step(act)        
        eval_reward_sum += reward       
        ob = ob_         
        if done:
            return  eval_reward_sum