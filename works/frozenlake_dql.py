# Deep Q Learning / Frozen Lake / Not Slippery / 4x4
# https://github.com/hamedmokazemi/DeepQLearning_FrozenLake_1/blob/main/main_4x4.py
# was anderes: https://gist.github.com/amoudgl/a04d68f273a22b4cb2d5c3261d61e91c
# simple neural network implementation of qlearning
import gym
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import pygame

# build environment
env = gym.make("FrozenLake-v1")
# env = wrappers.Monitor(env, '/tmp/frozenlake-qlearning', force=True)
n_obv = env.observation_space.n
n_acts = env.action_space.n

# initialization
learning_rate = 0.1
gamma = 0.99
train_episodes = 10000
episodes = 0
prev_state = env.reset()
episode_t = 0
e = 0.1

# create model
x = tf.compat.v1.placeholder(shape=[1, 16], dtype=tf.float32) #states
W = tf.compat.v1.Variable(tf.compat.v1.random_uniform([16, 4], 0, 0.1))
out = tf.matmul(x, W)
act = tf.argmax(out, 1)
t = tf.compat.v1.placeholder(shape=[1, 4], dtype=tf.float32) #actions
loss = tf.reduce_sum(tf.square(t - out))
train_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# start session
sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

rewards = 0
while episodes < train_episodes:
    episode_t += 1
    # take noisy action
    action, qvalues = sess.run([act, out], feed_dict={x: np.identity(16)[prev_state:prev_state + 1]})
    if (np.random.rand(1)) < e:
        action[0] = env.action_space.sample()
    next_state, rew, done, _ = env.step(action[0])

    # find targetQ values and update model
    qnext_values = sess.run([out], feed_dict={x: np.identity(16)[next_state:next_state + 1]})
    max_q = np.max(qnext_values)
    targetq = qvalues
    targetq[0, action[0]] = rew + gamma * max_q
    sess.run([train_step], feed_dict={x: np.identity(16)[prev_state:prev_state + 1], t: targetq})
    prev_state = next_state


    # episode finished
    if done:
        episodes += 1
        # decrease noise as number of episodes increases
        e = 1. / ((episodes / 50) + 10)
        prev_state = env.reset()
        #print("episode %d finished after %d timesteps, rew = %d" % (episodes, episode_t, rew))
        episode_t = 0
        rewards += rew

print(rewards)
print("Score over time: " + str(rewards / train_episodes))
