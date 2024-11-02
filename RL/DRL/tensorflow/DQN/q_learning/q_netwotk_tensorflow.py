import gym
import numpy as np
import matplotlib.pyplot as plt

import warnings  #tensorflow warnings https://github.com/tensorflow/tensorflow/issues/30427
warnings.filterwarnings('ignore',category=FutureWarning)

import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers

def train(env_name):
    #====== Edit =========
    EPISODE = 2000
    ALPHA = 0.5
    #====== Initial ======
    env=gym.make(env_name)

    N_ACT = env.action_space.n 
    N_OB  = env.observation_space.n 

    #===== NN =========
    model = keras.Sequential([
        layers.Dense(N_ACT, input_shape(N_OB,))
    ])

    model.compile(
        optimizer = Adam(learning_rate=ALPHA*1e-3),
        loss = "",
        metrics = ['accuracy']
    )

    model.fit(
        x = ob_vector,
        y = Qvalue_set,
        batch_size = 1,
        epochs = EPISODE,
        shuffle = True,
    )

    prediction=model.predict(
        x = ob_vector,
        batch_size = 1,
        verbose = 0
    )
    for i in range(EPISODE):
        env.render()

        ob_next, reward, done, info = env.step(act)
        
        if done:
            break

    env.close()

if __name__ == "__main__":
    env_name="FrozenLake-v0"
    train(env_name)