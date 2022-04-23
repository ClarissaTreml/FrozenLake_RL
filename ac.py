#https://keras-gym.readthedocs.io/en/stable/notebooks/frozen_lake/actor_critic.html
"""
import numpy as np
import gym
import keras

import keras_gym as km

# from tensorflow import keras
from werkzeug.datastructures import K

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)

action_space_size = env.action_space.n  # columns-> 4*4 -> 16
state_space_size = env.observation_space.n  # rows -> 4

num_episodes = 40000  # 10000  30000  50000
max_steps_per_episode = 100  # 100  200

learning_rate = 0.1  # Alpha # 0.1, 0.01, 0.2
discount_rate = 0.99  # Gamma # 0.99  0.1  1.0  0.5  0.8

exploration_rate = 1  # 0.7  1.0  0.99 # epsilon //mit 100 ->72% (bei 31000)
max_exploration_rate = 1.0  # 1.0   0.7  0.99
min_exploration_rate = 0.01  # 0.01   0.4  0.1
exploration_decay_rate = 0.001  # 0.001  # 0.01  0.05


class LinearFunc(km.FunctionApproximator):
    #linear function approximator (body only does one-hot encoding)

    def body(self, S):
        one_hot_encoding = keras.layers.Lambda(lambda x: K.one_hot(x, 16))
        return one_hot_encoding(S)


# define function approximators
function = LinearFunc(env, learning_rate)
pi = km.SoftmaxPolicy(function, update_strategy='ppo')
v = km.V(function, discount_rate, bootstrap_n=1)

# combine into one actor-critic
actor_critic = km.ActorCritic(pi, v)

rewards_all_episodes = []

# training:
for episode in range(num_episodes):
    state = env.reset()

    for step in range(max_steps_per_episode):
        action = pi(state, use_target_model=True)
        new_state, reward, done, info = env.step(action)

        # small incentive to keep moving
        if np.array_equal(new_state, s):
            r = -0.1

        actor_critic.update(s, a, r, done)

        if env.T % target_model_sync_period == 0:
            pi.sync_target_model(tau=1.0)

        if done:
            break

        s = new_state

    # run env one more time to render
    s = env.reset()
    env.render()

    for step in range(num_steps):

        # print individual action probabilities
        print("  v(s) = {:.3f}".format(v(s)))
        for i, p in enumerate(pi.dist_params(s)):
            print("  π({:s}|s) = {:.3f}".format(actions[i], p))

        a = pi.greedy(s)
        s, r, done, info = env.step(a)
        env.render()

        if done:
            break
"""