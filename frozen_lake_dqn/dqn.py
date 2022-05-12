# https://gist.github.com/sappelt/21f5fadcb2e58ed9f6c59f25d43b93cd#file-b_frozendeepq-ipynb
import numpy as np
import gym
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
from IPython.display import clear_output
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import random
import time

# import pygame

# env = gym.make('FrozenLake-v0')
#env = gym.make("FrozenLake-v1", is_slippery=False)
env = gym.make('FrozenLake-v1', desc=None,map_name="4x4", is_slippery=True)

n_acts = env.action_space.n  # 4
n_obv = env.observation_space.n  # 16

# Actions are left, up, right, down
# print(tf.version.VERSION)
#print("n_acts", n_acts)
# States are the 16 fields
#print("n_obv", n_obv)
# env.render()
# qtable = np.zeros((n_obv, n_acts))
# print(qtable)

# The inputs of the NN is the state space + next action
tf_input_size = n_obv  #16
# The output of the NN is the action space
tf_output_size = n_acts  #4
# The hidden layer size
tf_hidden_layer_size = (tf_input_size + tf_output_size) // 2
print("tf input size", tf_input_size)
print("tf output size", tf_output_size)
print("hidden layers", tf_hidden_layer_size)

# Reset the computational graph
tf.compat.v1.reset_default_graph()
# tf.reset_default_graph()

tf_inputs = tf.compat.v1.placeholder(tf.float32, [None, tf_input_size]) #16
tf_next_q = tf.compat.v1.placeholder(tf.float32, [None, tf_output_size]) #4

# Hidden Layers
tf_weights_1 = tf.compat.v1.get_variable("tf_weights_1", [tf_input_size, tf_hidden_layer_size], #16, 10
                                         initializer=tf.zeros_initializer)
tf_biases_1 = tf.compat.v1.get_variable("tf_biases_1", [tf_hidden_layer_size], initializer=tf.zeros_initializer)
tf_outputs_1 = tf.nn.relu(tf.matmul(tf_inputs, tf_weights_1) + tf_biases_1)

# Output
tf_weights_out = tf.compat.v1.get_variable("tf_weights_out", [tf_hidden_layer_size, tf_output_size],
                                           initializer=tf.zeros_initializer)
tf_biases_out = tf.compat.v1.get_variable("tf_biases_out", [tf_output_size], initializer=tf.zeros_initializer)

# Calculate the output layer
tf_outputs = tf.matmul(tf_outputs_1, tf_weights_out) + tf_biases_out

tf_action = tf.argmax(tf_outputs, 1)

# Calculate the loss by applying the softmax first
tf_loss = tf.reduce_sum(tf.square(tf_outputs - tf_next_q))

# Use adam optimizer (instead of GD) with a suboptimal learning rate of 0.1
tf_optimize = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(tf_loss)

sess = tf.compat.v1.InteractiveSession()
initializer = tf.compat.v1.global_variables_initializer()

sess.run(initializer)

total_episodes = 10000  # Total episodes
learning_rate = 0.9  # Learning rate                                #0.8 #0.1
max_steps = 99  # Max steps per episode
gamma = 0.95  # Discounting rate

# Exploration parameters
epsilon = 1.0  # Exploration rate
max_epsilon = 1.0  # Exploration probability at start               #1.0
min_epsilon = 0.5  # Minimum exploration probability                #0.01
decay_rate = 0.01  # Exponential decay rate for exploration prob    #0.005

# List of rewards
rewards = []

# 2 For life or until learning is stopped
for episode in range(total_episodes):
    # Reset the environment
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0

    for step in range(max_steps):
        # Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)

        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            # action = np.argmax(qtable[state,:])

            # Get the next action from our neural network
            action = sess.run([tf_action], feed_dict={tf_inputs: np.identity(16)[state:state + 1]})[0][0]
        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state
        # qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

        # Get the q-values for the current state
        old_q_values = sess.run([tf_outputs], feed_dict={tf_inputs: np.identity(16)[state:state + 1]})[0][0]
        old_value = old_q_values[action]

        # Get the q-values for the next state
        next_q_values = sess.run([tf_outputs], feed_dict={tf_inputs: np.identity(16)[new_state:new_state + 1]})[0][0]
        next_max = np.max(next_q_values)

        # Calculate the target value
        y_hat = reward + gamma * next_max

        # Set the new q-values (overwrite the old one)
        # q_table[current_state, action] = new_value
        new_q_values = old_q_values
        new_q_values[action] = y_hat

        # Train the NN
        # Run optimizer and calculate mean loss
        _, loss = sess.run([tf_optimize, tf_loss],
                           feed_dict={tf_inputs: np.identity(16)[state:state + 1],
                                      tf_next_q: new_q_values.reshape(1, 4)})

        # Our new state is state
        state = new_state

        # total_rewards += y_hat #war nicht default drin

        # If done (if we're dead) : finish episode
        if done == True:
            # print("Episode {} finished after {} timesteps".format(episode, step + 1))
            # Reduce epsilon (because we need less and less exploration)
            break

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    # rewards.append(total_rewards) #ist default drin
    # rewards.append(y_hat) #war nicht default drin
    rewards.append(reward)  # war nicht default drin

print("Score over time: " + str(sum(rewards) / total_episodes))
# Calculate and print the average reward per thousand episodes

rewards_per_thousand_episodes = np.split(np.array(rewards), total_episodes / 1000)
count = 1000
print("********Average reward per thousand episodes********\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r / 1000)))
    count += 1000

# Print updated Q-table
print("\n\n********Q-table********\n")
#print(qtable)

for episode in range(9997, 10000):
    # initialize new episode params
    state = env.reset()
    done = False
    print("*****EPISODE ", episode + 1, "*****\n\n\n\n")
    time.sleep(1)

    for step in range(max_steps):
        # Show current state of environment on screen
        # Choose action with highest Q-value for current state
        # Take new action
        clear_output(wait=True)
        env.render()
        time.sleep(0.3)

        #action = np.argmax(qtable[state, :])
        new_state, reward, done, info = env.step(action)

        if done:
            clear_output(wait=True)
            env.render()
            if reward == 1:
                # Agent reached the goal and won episode
                print("****You reached the goal!****")
                time.sleep(3)
            else:
                # Agent stepped in a hole and lost episode
                print("****You fell through a hole!****")
                time.sleep(3)
                clear_output(wait=True)

            print("Number of steps", step)
            break
        # Set new state
        state = new_state

env.close()
