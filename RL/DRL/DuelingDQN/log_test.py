# huge memo size
FRAME_END = 3e6

GAMMA  = 0.99

EPSILON      = 1.0
EPSILON_END  = 0.1
EPSILON_DECAY = 0.9999977

LEARNING_RATE = 0.0001

BATCH_SIZE  = 32

MEMORY_SIZE = 1_000_000
MEMORY_SAMPLE_START = 5_000

MODEL_UPDATE_STEP =  10_000
TRAIN_SKIP_STEP   =  4

ENV_NAME = "PongNoFrameskip-v4"
# 
RENDER = True
CLIP_FLAG = False
GIF_MAKE  = False
DDQN_FLAG = False # False for dqn
PER_FLAG  = False # False for normal replay
DOUBLE_FLAG = False # True for double q learning

# not happen , give up !!!!!
FRAME_END = 1e6

GAMMA  = 0.99

EPSILON      = 1.0
EPSILON_END  = 0.1
EPSILON_DECAY = 0.999995

LEARNING_RATE = 0.0001

BATCH_SIZE  = 32

MEMORY_SIZE = 100_000
MEMORY_SAMPLE_START = 5_000

MODEL_UPDATE_STEP =  10_000
TRAIN_SKIP_STEP   =  4

ENV_NAME = "PongNoFrameskip-v4"
# 
RENDER = True
CLIP_FLAG = False
GIF_MAKE  = False
DDQN_FLAG = False # False for dqn
PER_FLAG  = False # False for normal replay
DOUBLE_FLAG = False # True for double q learning
# %% dqn paper para time:16.30  date:1.1
# loss: may need to add more update step
# max q: normal seem
# ep decay too big
FRAME_END = 1e6

GAMMA  = 0.99

EPSILON      = 1.0
EPSILON_END  = 0.02
EPSILON_DECAY = 1-1e-5

LEARNING_RATE = 0.0001

BATCH_SIZE  = 32

MEMORY_SIZE = 100_000
MEMORY_SAMPLE_START = 5_000

MODEL_UPDATE_STEP =  10_000
TRAIN_SKIP_STEP   =  4

ENV_NAME = "PongNoFrameskip-v4"
# 
RENDER = True
CLIP_FLAG = False
GIF_MAKE  = False
DDQN_FLAG = False # False for dqn
PER_FLAG  = False # False for normal replay
DOUBLE_FLAG = False # True for double q learning
# %% dqn 1.1 15.20
# same para but new loss function
# better
# old loss is too big

# %% dqn 1.1 14.00
# check q value normal parameters
FRAME_END = 4e5

GAMMA  = 0.99

EPSILON      = 1.0
EPSILON_END  = 0.02
EPSILON_DECAY = 1-1e-5

LEARNING_RATE = 0.0001

BATCH_SIZE  = 32

MEMORY_SIZE = 10_000
MEMORY_SAMPLE_START = 1_000

MODEL_UPDATE_STEP =  1_000
TRAIN_SKIP_STEP   =  0

ENV_NAME = "PongNoFrameskip-v4"
# 
RENDER = True
CLIP_FLAG = False
GIF_MAKE  = False
DDQN_FLAG = False # False for dqn
PER_FLAG  = False # False for normal replay
DOUBLE_FLAG = False # True for double q learning
# %%
# a long trying one 13.13 12.31
# loss exploid
# so max
# no move at end
FRAME_END = 1e6

GAMMA  = 0.99

EPSILON      = 1.0
EPSILON_END  = 0.1
EPSILON_DECAY = 1-1e-6

LEARNING_RATE = 0.0001

BATCH_SIZE  = 32

MEMORY_SIZE = 10_000
MEMORY_SAMPLE_START = 1_000

MODEL_UPDATE_STEP =  1_000
TRAIN_SKIP_STEP   =  0

ENV_NAME = "PongNoFrameskip-v4"
# 
RENDER = True
CLIP_FLAG = False
GIF_MAKE  = False
DDQN_FLAG = False # False for dqn
PER_FLAG  = False # False for normal replay
DOUBLE_FLAG = False # True for double q learning

# %%   dqn per
# still not learning, but loss final and max q normal
# medium decay
# long update 
# long memo
FRAME_END = 4e5

GAMMA  = 0.99

EPSILON      = 1.0
EPSILON_END  = 0.01
EPSILON_DECAY = 1-1e-5

LEARNING_RATE = 0.0001

BATCH_SIZE  = 32

MEMORY_SIZE = 10_000
MEMORY_SAMPLE_START = 1_000

MODEL_UPDATE_STEP =  1_000
TRAIN_SKIP_STEP   =  0

ENV_NAME = "PongNoFrameskip-v4"
# 
RENDER = True
CLIP_FLAG = False
GIF_MAKE  = False
DDQN_FLAG = False # False for dqn
PER_FLAG  = True # False for normal replay
DOUBLE_FLAG = False # True for double q learning
# %% dqn logn train failed 12.31
# loss too big
FRAME_END = 5e5

GAMMA  = 0.99

EPSILON      = 1.0
EPSILON_END  = 0.01
EPSILON_DECAY = 1-1e-6

LEARNING_RATE = 0.0001

BATCH_SIZE  = 32

MEMORY_SIZE = 50_000
MEMORY_SAMPLE_START = 1_000

MODEL_UPDATE_STEP =  1_000
TRAIN_SKIP_STEP   =  0

ENV_NAME = "PongNoFrameskip-v4"
# 
RENDER = True
CLIP_FLAG = False
GIF_MAKE  = False
DDQN_FLAG = False # False for dqn
PER_FLAG  = False # False for normal replay
DOUBLE_FLAG = False # True for double q learning
# dqn big memo 22.00 12.30
# not help much bad loss
FRAME_END = 3.5e5

GAMMA  = 0.99

EPSILON      = 1.0
EPSILON_END  = 0.1
EPSILON_DECAY = 0.99999

LEARNING_RATE = 0.0001

BATCH_SIZE  = 32

MEMORY_SIZE = 50_000
MEMORY_SAMPLE_START = 1_000

MODEL_UPDATE_STEP =  1_000
TRAIN_SKIP_STEP   =  0

ENV_NAME = "PongNoFrameskip-v4"
# 
RENDER = True
CLIP_FLAG = False
GIF_MAKE  = False
DDQN_FLAG = False # False for dqn
PER_FLAG  = False # False for normal replay
DOUBLE_FLAG = False # True for double q learning
#dqn fast update 19.40 12.30
# so bad loss
FRAME_END = 4e5

GAMMA  = 0.99

EPSILON      = 1.0
EPSILON_END  = 0.1
EPSILON_DECAY = 0.99999

LEARNING_RATE = 0.0001

BATCH_SIZE  = 32

MEMORY_SIZE = 10_000
MEMORY_SAMPLE_START = 1_000

MODEL_UPDATE_STEP =  500
TRAIN_SKIP_STEP   =  0

ENV_NAME = "PongNoFrameskip-v4"
# 
RENDER = True
CLIP_FLAG = False
GIF_MAKE  = False
DDQN_FLAG = False # False for dqn
PER_FLAG  = False # False for normal replay
DOUBLE_FLAG = False # True for double q learning
# dqn 19.00 12.30
# not help
CLIP = True
FRAME_END = 4e5

GAMMA  = 0.99

EPSILON      = 1.0
EPSILON_END  = 0.02
EPSILON_DECAY = 0.999985

LEARNING_RATE = 0.0001

BATCH_SIZE  = 32

MEMORY_SIZE = 20_000
MEMORY_SAMPLE_START = 1_000

MODEL_UPDATE_STEP =  5_000
TRAIN_SKIP_STEP   =  0

ENV_NAME = "PongNoFrameskip-v4"

# dqn 17.30 12.30
# learn at first then stuck
FRAME_END = 4e5

GAMMA  = 0.99

EPSILON      = 1.0
EPSILON_END  = 0.02
EPSILON_DECAY = 0.999985

LEARNING_RATE = 0.0001

BATCH_SIZE  = 32

MEMORY_SIZE = 20_000
MEMORY_SAMPLE_START = 1_000

MODEL_UPDATE_STEP =  5_000
TRAIN_SKIP_STEP   =  0

ENV_NAME = "PongNoFrameskip-v4"
# dqn 10.13 12.30
FRAME_END = 4e5

GAMMA  = 0.99

EPSILON      = 1.0
EPSILON_END  = 0.02
EPSILON_DECAY = 0.999985

LEARNING_RATE = 0.0001

BATCH_SIZE  = 32

MEMORY_SIZE = 20_000
MEMORY_SAMPLE_START = 1_000

MODEL_UPDATE_STEP =  20
TRAIN_SKIP_STEP   =  4

ENV_NAME = "PongNoFrameskip-v4"
# per_dqn 22.30 12.29
# max_q exploid !!!
FRAME_END = 3e5

GAMMA  = 0.99

EPSILON      = 1.0
EPSILON_END  = 0.02
EPSILON_DECAY = 0.99998

LEARNING_RATE = 0.0001

BATCH_SIZE  = 32

MEMORY_SIZE = 20_000
MEMORY_SAMPLE_START = 1_000

MODEL_UPDATE_STEP =  1_000
TRAIN_SKIP_STEP   =  0

RENDER = False
GIF_MAKE = False

ENV_NAME = "PongNoFrameskip-v4"

# per_dqn 22.00 12.29
# more unstable
FRAME_END = 3e5

GAMMA  = 0.99

EPSILON      = 1.0
EPSILON_END  = 0.02
EPSILON_DECAY = 0.99998

LEARNING_RATE = 0.0001

BATCH_SIZE  = 32

MEMORY_SIZE = 10_000
MEMORY_SAMPLE_START = 1_000

MODEL_UPDATE_STEP =  2_000
TRAIN_SKIP_STEP   =  4

RENDER = False
GIF_MAKE = False

ENV_NAME = "PongNoFrameskip-v4"

# per_dqn 21.00 12.29
"""
FRAME_END:300000.0
EPSILON:1.0, EPSILON_END:0.02, EPSILON_DECAY:0.99998
GAMMA: 0.99,  LEARNING_RATE:0.0001 
MEMORY_SIZE:10000 ,MEMORY_SAMPLE_START:1000
MODEL_UPDATE_STEP:1000, TRAIN_SKIP_STEP:4
BATCH_SIZE:32
"""
FRAME_END = 3e5

GAMMA  = 0.99

EPSILON      = 1.0
EPSILON_END  = 0.02
EPSILON_DECAY = 0.99998

LEARNING_RATE = 0.0001

BATCH_SIZE  = 32

MEMORY_SIZE = 10_000
MEMORY_SAMPLE_START = 1_000

MODEL_UPDATE_STEP =  1_000
TRAIN_SKIP_STEP   =  4

RENDER = False
GIF_MAKE = False

ENV_NAME = "PongNoFrameskip-v4"