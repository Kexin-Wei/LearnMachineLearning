# n-step 
RENDER_FLAG      = False
TENSORBOARD_FLAG = False
EPOCHS = 2000
GAMMA  = 0.99
N_STEP = 4
alpha = 1e-5
fc1   = 1024
fc2   = 256
# n-step not help 14.32 1.07
# maybe average loss?
RENDER_FLAG      = False
TENSORBOARD_FLAG = False
EPOCHS = 2000
GAMMA  = 0.99
N_STEP = 4
alpha = 1e-3
fc1   = 1024
fc2   = 256
# n-step 14.20 1.07
# better??
# maybe smaller??
RENDER_FLAG      = False
TENSORBOARD_FLAG = False
EPOCHS = 2000
GAMMA  = 0.99
N_STEP = 4
alpha = 1e-4
fc1   = 1024
fc2   = 256
# n-step 14.13 1.07
# worse???
# maybe alpha too small??
RENDER_FLAG      = False
TENSORBOARD_FLAG = False
EPOCHS = 2000
GAMMA  = 0.99
N_STEP = 4
alpha = 1e-8
fc1   = 1024
fc2   = 256
# n-step not converge 14.02 1.07
# may be the alpha too big?
RENDER_FLAG      = False
TENSORBOARD_FLAG = False
EPOCHS = 2000
GAMMA  = 0.99
N_STEP = 4
alpha = 1e-7
fc1   = 1024
fc2   = 256
#1.07 13.20
# LunarLander 
# group compare test
# lr<1e-6

# change to cartploe with 20_000 1.06
# good, converge slow
# better but still has low point
EPOCHS = 2000
agent = Agent(ALPHA=1e-5, IN_DIMS=N_OB, N_ACT=N_ACT,
                GAMMA = 0.99, FC1_DIMS = 2048, FC2_DIMS = 128)

#ac one step 17.02
# not good
EPOCHS = 2000
agent = Agent(ALPHA=1e-5, IN_DIMS=N_OB, N_ACT=N_ACT,
                GAMMA = 0.99, FC1_DIMS = 512, FC2_DIMS = 128)
