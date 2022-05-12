# https://gist.github.com/programming-datascience/d8b96346e347b0b6942e16a33e64039c
# https://www.youtube.com/watch?v=OxV2eUqCmzw

# RuntimeError: mat1 and mat2 shapes cannot be multiplied (33600x3 and 12x64)  --> SpaceInvaders, Breakout
# TypeError: expected np.ndarray (got int) --> FrozenLake
# CartPoleEnv works (FrozenLake and SpaceInvaders and Breakout not)

import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import pygame

# Importing PyTorch here

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

#env = gym.make('FrozenLake-v1')
#env = gym.make('ALE/SpaceInvaders-v5', render_mode='human')
env = gym.make('CartPole-v1')  # We make the Cartpole environment here
#env = gym.make('ALE/Breakout-v5', render_mode='human')

print("There are {} actions".format(env.action_space.n))


# You can move either left or right to balance the pole
# Lets implement the Actor critic network
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(4, 128)  # 4 because there are 4 parameters as the observation space (4, 128)
        self.actor = nn.Linear(128, 2)  # 2 for the number of actions       (128, 2)
        self.critic = nn.Linear(128, 1)  # Critic is always 1               (128, 1)
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.actor(x), dim=-1)
        state_values = self.critic(x)
        return action_prob, state_values


def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()


# In this function, we decide whehter we want the block to move left or right,based on what the model decided

def finish_episode():
    # We calculate the losses and perform backprop in this function
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    returns = []

    for r in model.rewards[::-1]:
        R = r + 0.99 * R  # 0.99 is our gamma number
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        policy_losses.append(-log_prob * advantage)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    loss.backward()
    optimizer.step()

    del model.rewards[:]
    del model.saved_actions[:]


model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()


def train():
    running_reward = 10
    for i_episode in count():  # We need around this much episodes
        state = env.reset()
        ep_reward = 0
        for t in range(1, 10000):
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % 10 == 0:  # We will print some things out
            print("Episode {}\tLast Reward: {:.2f}\tAverage reward: {:.2f}".format(
                i_episode, ep_reward, running_reward
            ))
        if running_reward > env.spec.reward_threshold:
            print("Solved, running reward is now {} and the last episode runs to {} time steps".format(
                running_reward, t
            ))
            break
            # This means that we solved cartpole and training is complete

train()

# There. we finished
# Lets see it in action
done = False
cnt = 0

observation = env.reset()

while True:
    cnt += 1
    env.render()
    action = select_action(observation)
    observation, reward, done, _ = env.step(action)
    # Lets see how long it lasts until failing

print("Game lasted {cnt} moves")
