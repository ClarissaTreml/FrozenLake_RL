# pip install ale-py
# pip install torch gym
# pip install gym[atari]
# pip install autorom
# AutoROM

# https://www.youtube.com/watch?v=NP8pXZdU-5U
# PyTorch Documentation:
# https://pytorch.org/docs/stable/index.html
# pip install torch gym #installs pytorch and opens ai gym
# pip install ale-py

#works for atari

from torch import nn
import torch
import gym

from collections import deque
import itertools
import numpy as np
import random

# Parameters
# Are different for every single environment:
# See Research Paper page 10 for different parameters: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
GAMMA = 0.99  # discount rate for computing our temporal difference target
BATCH_SIZE = 32  # how many transitions we're going to sample from the replay buffer
BUFFER_SIZE = 50000  # maximum number of transitions we're going to store before overwriting old transitions
MIN_REPLAY_SIZE = 1000  # how many transitions we want in the replay buffer before we start computing gradients and doing training
EPSILON_START = 1.0  # starting value for our epsilon
EPSILON_END = 0.02  # ending value of our epsilon
EPSILON_DECAY = 10000  # decay period which the epsilon will linearly anneal from "epsilon start" to "epsilon end" with "epsilon decay" many steps
TARGET_UPDATE_FREQ = 1000  # number of steps where we set the target parameters equal to the online parameters


# create network class which inherits from nn.module
class Network(nn.Module):
    def __init__(self, env):  # initialize the class calling the superclass
        super().__init__()

        in_features = int(np.prod(
            env.observation_space.shape))  # compute the number of inputs to this network by taking the product of
        # the observation space shape
        # shape for dimensions (cartpole = 1 dimension), (atari = 3 dimensional shape for the images)
        # this is: how many neurons are in the input layer of our neural network

        self.net = nn.Sequential(  # standard two layer sequential linear network
            nn.Linear(in_features, 64),  # with 64 hidden units
            nn.Tanh(),  # seperated by a tan h lin non-linearity
            nn.Linear(64,
                      env.action_space.n))  # number of output units in our network is going to be equal to the number
        # of possible actions that the agent can take inside that environment which is this action space

    # environments with continuous action spaces  require different algorithms
    def forward(self, x):  # do forward function
        return self.net(x)  # which every single nn.module is required to implement

    #select an action
    def act(self, obs):  # act function (placeholder)
        obs_t = torch.as_tensor(obs, dtype=torch.float32)  # we turn obs into a pi torch tensor
        q_values = self(obs_t.unsqueeze(0))  # compute the q values for this specific observations, for every single possible action the agent can take
        # unsqueeze(0): 0 because there is no Batch Dimension. Basically we want to create a fake Batch dimension.

        max_q_index = torch.argmax(q_values, dim=1)[0] #getting action with highest q value
        action = max_q_index.detach().item() #we turn max_q_index into an integer

        return action # the action indicee (a number between 0 and 1 minus the number of actions


env = gym.make('ALE/SpaceInvaders-v5', render_mode='human')  # create our environment
#env = gym.make('ALE/Breakout-v5', render_mode='human')

replay_buffer = deque(
    maxlen=BUFFER_SIZE)  # create replay buffer which is the standard deck from python the max length buffer size
rew_buffer = deque([0.0],
                   maxlen=100)  # create reward buffer were we store the rewards earned by our agent in a single
# episode (to track the improvement of the agent)

episode_reward = 0.0  # reward for this specific episode

online_net = Network(env)  # create our network (online net)
target_net = Network(env)  # create our target (target net)

target_net.load_state_dict(
    online_net.state_dict())  # set the target net parameters equal to the online network parameters

optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4)
# create optimizer with torch.optim.Adam() method: optimize the online.net parameters with learning rate.
# TODO make learning rate a constant!

# initialize replay buffer
obs = env.reset()

for _ in range(MIN_REPLAY_SIZE):  # loop min replay size times
    action = env.action_space.sample()  # select a random action by calling the action space.sample method

    new_obs, rew, done, _ = env.step(
        action)  # step the environment with that action -> result:
    # new observation (which is the state of the environment)
    #   rew (reward returned by the environment for that action)
    #  done (whether the env episode is done, the env needs to be reset)
    #  info (info dictionary which we are not using) -> replace with _
    transition = (obs, action, rew, done, new_obs)  # create transition tuple with obs, action, ...
    replay_buffer.append(
        transition)  # stick the transition inside our replay buffer
    # (now we have that experience inside our replay buffer which we can use to train on later)
    obs = new_obs  # observation is equal to new observation

    if done:  # if env needs to be reset
        obs = env.reset()  # reset it

# Main Training Loop
# using the epsilon greedy policy
# we reset the environment and get the observation
obs = env.reset()

for step in itertools.count():  # like a while true loop
    # we need to select an action to take in the environment
    # first we need to compute epsilon, since it interpolates between epsilon start and end value over epsilon decay steps
    # np.interp is a numpy functin which does the computing
    # it starts out at epsilon start and ends up at epsilon end after epsilon decay steps and for every step after that it will still be epsilon end
    # this is done to facilitate exploration by the agent in the environment
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START,
                                                   EPSILON_END])  # start is 1.0 and end is 0.02 --> we go from 100 random actions to 2 random actions

    rnd_sample = random.random()
    # we get a random sample and if it is <= epsilon we take a random action using the axis space.sample method again!!!!!
    if rnd_sample <= epsilon:
        action = env.action_space.sample()
    else:  # we need to intelligently select action using our network for that we call online_net.act and pass in the observation
        action = online_net.act(obs)
    # copied from the initialized replay buffer section
    new_obs, rew, done, _ = env.step(
        action)  # we are stepping the environment and getting all the new data, the reward, the new observation
    transition = (obs, action, rew, done, new_obs)  # setting up the transition tuple
    replay_buffer.append(transition)  # putting the transition tuple in the replay buffer
    obs = new_obs  # allways setting obs to new obs

    episode_reward += rew  # we add the reward for this step to the episode reward

    if done:
        obs = env.reset()  # resetting the evironment if it needs to be reset

        rew_buffer.append(episode_reward)  # we need to append the episode reward to the reward buffer
        episode_reward = 0.0  # and reset the episode reward

    # After solved, watch it play
    if len(rew_buffer) >= 100:  # if we have at least 100 episodes
        if np.mean(rew_buffer) >= 195:  # if the average reward_buffer is at least a 195
            while True:  # Infinite Loop to ignore rest of Code, because max score 'CartPole-v0' is 200
                action = online_net.act(obs)

                obs, _, done, _ = env.step(action)  # _ ignore values
                env.render()
                if done:
                    env.reset()

    # Start Gradient Step
    transitions = random.sample(replay_buffer, BATCH_SIZE) # sample batch size number of random transitions from our replay buffer that we added in earlier
    # all transitions are structured as tuple but we need each element individually in their own arrays
    # get each observation and put it in a list from the transitions we sampled
    obses = np.asarray([t[0] for t in transitions])
    actions = np.asarray([t[1] for t in transitions])
    rews = np.asarray([t[2] for t in transitions])
    dones = np.asarray([t[3] for t in transitions])
    new_obses = np.asarray([t[4] for t in transitions])

    # we convert the aboce arrays into a pi torch tensor
    obses_t = torch.as_tensor(obses, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
    dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
    new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)
    # unsqueeze(-1) for batch dimension. -1 because we already have a BATCH_SIZE number of actions and that
    # is the Batch Dimension. We want to put these values inside its own dimension.

    # Compute targets for our loss function
    # we have a set of q values for each observation and each observation is the batch dimension and the q values are dimension one
    target_q_values = target_net(new_obses_t) # get target q values for the next/new ops's
    # for each of these new observations we have a set of q values but we need to collapse this down to one being the highest q value per observation
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0] # get the maximum value in dimension one and discard the rest and keep that
    # dimension around even thought there is only one vlaue in it
    # we do the 0 index because max returns a tuple where the first element is the highest values and the second element are the index of those values
    #which is equivalent to argmax

    #Compute targets
    targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values
    # Rewards plus Gamma times max-target-q-values, but if (1 - dones_t(terminal state)) is 0,
    # everything will be 0 except Rewards.
    # See Research Paper page 7 for equation: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

    # Compute Loss
    q_values = online_net(obses_t)
    # a set of q-values for each observation
    # here we don't want to get the max-value:

    action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)
    # we want the q values for the action we actually took.
    # torch.gather() applies the index and dimension to all the elements of the input.

    loss = nn.functional.smooth_l1_loss(action_q_values, targets)
    # TorchDoc: loss = nn.functional.smooth_l1_loss(action_q_values, targets)
    # Function that uses a squared term if the absolute element-wise error falls below beta and an L1 term otherwise.
    # See for more Information:
    # https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html#torch.nn.SmoothL1Loss

    # Gradient Descent
    optimizer.zero_grad()  # zero out the gradients
    loss.backward()  # to compute gradients
    optimizer.step()  # to apply gradients

    # Update Target Network
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())
        # load_state_dict() function will set the parameters of the target_net
        # to the same values as those in the online_net

    # Logging
    # to see if things are improving
    if step % 1000 == 0:
        print()
        print('Step', step)  # print put current step
        print('Avg Rew',
              np.mean(rew_buffer))  # print out the average reward from reward_buffer over the last 100 episodes
