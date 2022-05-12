# https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py
import numpy as np
import gym
import random
import time
from IPython.display import clear_output

# env = gym.make("FrozenLake-v1")
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)

action_space_size = env.action_space.n  # columns-> 4*4 -> 16
state_space_size = env.observation_space.n  # rows -> 4

"""
 ### Observation Space
    The observation is a value representing the agent's current position as
    current_row * nrows + current_col (where both the row and col start at 0).
    For example, the goal position in the 4x4 map can be calculated as follows: 3 * 4 + 3 = 15.
    The number of possible observations is dependent on the size of the map.
    For example, the 4x4 map has 16 possible observations.
"""

q_table = np.zeros((state_space_size, action_space_size))
print(q_table)

num_episodes = 40000  # 10000  30000  50000
max_steps_per_episode = 100  # 100  200

learning_rate = 0.1  # Alpha # 0.1, 0.01, 0.2
discount_rate = 0.99  # Gamma # 0.99  0.1  1.0  0.5  0.8

exploration_rate = 1  # 0.7  1.0  0.99 # epsilon //mit 100 ->72% (bei 31000)
max_exploration_rate = 1.0  # 1.0   0.7  0.99
min_exploration_rate = 0.01  # 0.01   0.4  0.1
exploration_decay_rate = 0.001  # 0.001  # 0.01  0.05

rewards_all_episodes = []

# Q-learning algorithm
# This first for-loop contains everything that happens within a single episode
for episode in range(num_episodes):
    # initialize new episode params
    state = env.reset()  # For each episode, we're going to first reset the state of the environment back to the
    # starting state.

    done = False  # The done variable just keeps track of whether or not our episode is finished
    rewards_current_episode = 0  # Then, we need to keep track of the rewards within the current episode as well, so we
    # set rewards_current_episode to 0 since we start out with no rewards at the beginning of each episode.

    # This second nested loop contains everything that happens for a single time-step
    for step in range(max_steps_per_episode):
        # Exploration-exploitation trade-off
        # Take new action
        # Update Q-table
        # Set new state
        # Add new reward

        # Exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(0, 1)  # For each time-step within an episode, we set our
        # print("Exploration Rate: ", exploration_rate_threshold)
        # exploration_rate_threshold to a random number between 0 and 1. This will be used to determine whether our
        # agent will explore or exploit the environment in this time-step

        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])  # exploit the environment and choose the action that has the highest
            # print("Exploit action: ", action)
            # Q-value in the Q-table for the current state. | exploitation = act of exploiting info thats already known
            # about the env in order to maximize the return
            # qtable[state,:] : all the actions we can take from current state
        else:
            action = env.action_space.sample()  # explore the environment and sample an action randomly | explore =
            # exploring env to find out info about it
            # print("Explore action: ", action)

        new_state, reward, done, info = env.step(action)

        # print("new state: ", new_state)
        # print("reward: ", reward)
        # print("done: ", done)
        # print("info: ", info)

        # Update Q-table for Q(s,a)
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
                                 learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        # print("q table: ", q_table)
        state = new_state
        # print("state", state)
        rewards_current_episode += reward
        # print("Rewards Current Episode", rewards_current_episode)

        if done == True:
            # print("EPISODE DONE **********************************************")
            break

    # Exploration rate decay
    # Add current episode reward to total rewards list

    # Exploration rate decay

    rewards_all_episodes.append(rewards_current_episode)

# Calculate and print the average reward per thousand episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 1000)
count = 1000
print("********Average reward per thousand episodes********\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r / 1000)))
    count += 1000

# Print updated Q-table
print("\n\n********Q-table********\n")
print(q_table)
"""
for episode in range(3):
    # initialize new episode params
    state = env.reset()
    done = False
    print("*****EPISODE ", episode + 1, "*****\n\n\n\n")
    time.sleep(1)

    for step in range(max_steps_per_episode):
        # Show current state of environment on screen
        # Choose action with highest Q-value for current state
        # Take new action
        clear_output(wait=True)
        env.render()
        time.sleep(0.3)

        action = np.argmax(q_table[state, :])
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
"""
env.close()
