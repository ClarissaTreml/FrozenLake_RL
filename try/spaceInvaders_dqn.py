# https://www.youtube.com/watch?v=hCeJeq8U0lo
# https://github.com/nicknochnack/KerasRL-OpenAI-Atari-SpaceInvadersv0/blob/main/Space%20Invaders%20Walkthrough.ipynb
# pip install tensorflow gym keras-rl2 gym[atari]
# bei amina ja, clarissa nein, jasmine nein

import epochs as epochs
import gym
import random
import numpy as np
import tensorflow as tf
from keras.optimizer_v1 import adam
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Convolution2D
# from tensorflow.python.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# env = gym.make('CartPole-v0')
# env = gym.make('ALE/SpaceInvaders-v5', render_mode='human')
env = gym.make('ALE/SpaceInvaders-v5', render_mode='human')
# env = gym.make('ALE/Breakout-v5', render_mode='human')
height, width, channels = env.observation_space.shape
actions = env.action_space.n

env.unwrapped.get_action_meanings()

episodes = 5
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        # env.render()
        action = env.action_space.sample()
        #action = random.choice([0, 1, 2, 3, 4, 5])
        n_state, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()


def build_model(height, width, channels, actions):
    model = Sequential()
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(3, height, width, channels)))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

    del model


model = build_model(height, width, channels, actions)
model.summary()


def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.2,
                                  nb_steps=10000)
    memory = SequentialMemory(limit=1000, window_length=3)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   enable_dueling_network=True, dueling_type='avg',
                   nb_actions=actions, nb_steps_warmup=1000
                   )
    return dqn


dqn = build_agent(model, actions)
dqn.compile(optimizer=adam, lr=1e-4)

dqn.fit(env, nb_steps=10000, visualize=False, verbose=2)

scores = dqn.test(env, nb_episodes=10, visualize=True)
print(np.mean(scores.history['episode_reward']))

dqn.save_weights('SavedWeights/10k-Fast/dqn_weights.h5f')
del model, dqn
dqn.load_weights('SavedWeights/1m/dqn_weights.h5f')
