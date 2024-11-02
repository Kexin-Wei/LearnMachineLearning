# %%
import argparse
def get_args():
    parser = argparse.ArgumentParser(description='PPO Hyperparameters Adjust')
    parser.add_argument("-ep","--EPOCHS",default=2000,
                                    type=int,
                                    help="Whole program epochs")
    parser.add_argument("--EVAL_FREQ",default=10,
                                    type=int,
                                    help="Trainning to evaluate model per x epochs")
    """
    parser.add_argument("--TRIAN_PER_EP",default=10,
                                    type=int,
                                    help="Trainning to update model x times per epoch")
    
    parser.add_argument("--STEPS_PER_EP",default=2048,
                                    type=int,
                                    help="Collect constant steps per ep, controled by ONE_EP_FLAG")
    
    parser.add_argument("--NUM_WORKERS",default=1,
                                    type=int,
                                    help="Parallel worker number to collect experience, total collections equal STEPS_PER_EP ")
    
    parser.add_argument("--ONE_TRAJ_FLAG",default=False,
                                    action="store_true",
                                    help="Flag to collect steps per ep until env ends, Fasle to collect constant steps, controlled by STEPS_PER_EP ")
    """
    parser.add_argument("-la","--lr_actor",default=1e-4,
                                    type=float,
                                    help="Learning rate of actor model under Adam")
    
    parser.add_argument("-lc","--lr_critic",default=1e-3,
                                    type=float,
                                    help="Learning rate of critic model under Adam")
    
    """parser.add_argument("--EPSILON",default=0.2,
                                    type=float,
                                    help="Loss clip parameter: rs(1-epsilon,1+epsilon)*Advantage")"""
    
    parser.add_argument("--GAMMA",default=0.99,
                                    type=float,
                                    help="TD error parameter: r + GAMMA * V(s') - V(s)")
    
    parser.add_argument("-t","--TAU",default=0.001,
                                    type=float,
                                    help="Parameter for target network update: target = tau*trained + (1-tau)*target")
    
    """ parser.add_argument("--LAMBDA",default=0.97,
                                    type=float,
                                    help="GAE parameter: advantage = sum( LAMBDA * GAMMA * td_error)")"""
    
    parser.add_argument("--fc_n",default=[400,300],
                                    nargs="+",
                                    type=int,
                                    help="Hidden layer number of Network")
    
    """parser.add_argument("--COEF_ENTROPY",default=0.1,
                                    type=float,
                                    help="Coefficience of entropy loss")
    
    parser.add_argument("--COEF_VALUE",default=0.5,
                                    type=float,
                                    help="Coefficience of value loss")"""
    
    parser.add_argument("-ms","--MEMORY_SIZE",default=int(1e6),
                                    type=int,
                                    help="Memory size for replay deque")
    
    parser.add_argument("--BATCH_SIZE",default=64,
                                    type=int,
                                    help="Batchsize of experience sample when trainning")
    
    
    """parser.add_argument("-n","--NORMALIZE",default=False,
                                    action="store_true",
                                    help="Flag to normalize observations for model")"""
    
    parser.add_argument("--DEVICE",default=None,
                                    choices=['cpu','cuda:0'],
                                    help="Specify using 'cpu' or 'gpu', otherwise use gpu is capable")
    
    """parser.add_argument("--TIMESTEP",default=0.05,
                                    type=float,
                                    help="Vrep timestep")
    
    parser.add_argument("--MAX_TIMESTEPS",default=3.0,
                                    type=float,
                                    help="Max timestep per run in vrep")
    
    parser.add_argument("--PORT",default=20010,
                                    type=int,
                                    help="Port for first env in vrep")
    
    parser.add_argument("--HIDE",default=True,
                                    action="store_false",
                                    help="Hide trajectory collection vrep envs")
    
    parser.add_argument("-r","--RANDOM_START",default=False,
                                    action="store_true",
                                    help="Set the target position random at simulation start")
    
    parser.add_argument("--BOUNDED",default=False,
                                    action="store_true",
                                    help="Bound the workspace of robot in vrep")"""
                            
    
    parser.add_argument("--env_name",default="LunarLanderContinuous-v2",
                                    choices=["LunarLanderContinuous-v2",
                                             "MountainCarContinuous-v0",
                                             "BipedalWalkerHardcore-v3"],
                                    help="Choose name of environment")
    
    args = parser.parse_args()
    
    return args

# %%
import platform
import os
import datetime
def dir_maker(args,FILENAME):
    comment = (f"lr_actor_{args.lr_actor}_lr_critic_{args.lr_critic}"
               f"_ep_{args.EPOCHS}_eval_{args.EVAL_FREQ}")
    
    for i in range(len(args.fc_n)):        
        comment += f"_fc{i+1}_{args.fc_n[i]}"
    
    comment += f"_{datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')}"
    
    OS = "mac" if platform.system() == "Darwin" else "linux"
    PARENT_DIR = f"test_{OS}_{FILENAME}_{args.env_name}"
    DIR = os.path.join(PARENT_DIR,comment)
    
    try:
        os.makedirs(DIR)
    except:
        print(f"Failed to open folder {DIR}")    
        
    return DIR,comment