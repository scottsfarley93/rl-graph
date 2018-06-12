from __future__ import print_function

import gym
import math
import random
import numpy as np
import matplotlib
from collections import namedtuple
from itertools import count
import time


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

import gym_graph
from visdom import Visdom


## setup viz
viz = Visdom()
startup_sec = 2
while not viz.check_connection() and startup_sec > 0:
    time.sleep(0.1)
    startup_sec -= 0.1
assert viz.check_connection(), 'No visualization connection could be formed quickly. Is the server running? '



env = gym.make("simple-static-graph-v0").unwrapped


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

state_size = len(env.reset())
nb_actions = env.action_space.n


class DQN(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(DQN, self).__init__()
        self.num_actions = n_output
        self.dense1 = torch.nn.Linear(n_feature, 1024)   # hidden layer
        self.dense2 = torch.nn.Linear(1024, 2048)   # hidden layer
        self.fc1_adv = nn.Linear(in_features=2048, out_features=512)
        self.fc1_val = nn.Linear(in_features=2048, out_features=512)
        #
        self.fc2_adv = nn.Linear(in_features=512, out_features=self.num_actions)
        self.fc2_val = nn.Linear(in_features=512, out_features=1)
        #
        self.relu = nn.ReLU()
        self.out = self.calc_out

    def calc_out(self, val, adv, x):
        return val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)


    def forward(self, x):
        x = self.relu(self.dense1(x))
        # x = self.relu(self.dense2(x))
        # # x = self.relu(self.dense2(x))
        # print(x.shape)
        # x = x.view(x.size(0), -1)
        # print(x.shape)
        #
        adv = self.relu(self.fc1_adv(x))
        val = self.relu(self.fc1_val(x))
        #
        adv = self.fc2_adv(adv)
        val = self.fc2_val(val)
        x = self.out(val, adv, x)
        return x

device = "cpu"

BATCH_SIZE = 128
GAMMA = 0.95
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 50
TARGET_UPDATE = 10

ROLLING_MEAN_100 = 0

policy_net = DQN(state_size, nb_actions)
target_net = DQN(state_size, nb_actions)
if os.path.exists("./dqn_graph.pt"):
    try:
        print( "Found existing dqn model. Loading from checkpoint.")
        policy_net.load_state_dict(torch.load('./dqn_graph.pt'))
    except:
        print ("Failed to load existing model checkpoint file. Starting from scratch.")
        pass
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    s = state.unsqueeze(0).to(device)
    if sample > eps_threshold:
        with torch.no_grad():
            action =  policy_net(s)[0].max(0)[1]
            view = action.view(1, 1)
            return view
    else:
        return torch.tensor([[random.randrange(nb_actions)]], device=device, dtype=torch.long)

episode_durations = []
episode_scores = []
mean_scores = []

score_win = None
step_win = None
scatter_win = None
text_win = None
mean_score_win = None
loss_win = None

def plot():
    X = np.array(range(0, len(episode_scores)))
    scores = np.array(episode_scores)
    steps = np.array(episode_steps)
    mean_100 = np.array(mean_scores)
    losses_np = np.array(losses)


    global score_win
    global step_win
    global scatter_win
    global mean_score_win
    global loss_win

    if not score_win:
        score_win =  viz.line(
            Y=scores,
            X=X,
            opts = dict(
                title="Episode Reward",
                xlabel="Episode",
                ylabel="Reward"
            )
        )
    else:
        viz.line(Y=scores, X=X, update="replace", win=score_win)

    if not step_win:
        step_win =  viz.line(
            Y=steps,
            X=X,
            opts=dict(
                title="Episode Steps",
                xlabel="Episode",
                ylabel="# Steps"
            )

        )
    else:
        viz.line(Y=steps, X=X, update="replace", win=step_win)


    if len(losses) > 0:
        if not loss_win:
            loss_win =  viz.line(
                Y=losses_np,
                X=np.array(range(0, len(losses))),
                opts=dict(
                    title="Loss",
                    xlabel="Step",
                    ylabel="Loss Value",
                )
            )
        else:
            viz.line(Y=losses_np, X=np.array(range(0, len(losses))), update="replace", win=loss_win)

    # if not scatter_win:
    #     scatter_win =  viz.scatter(
    #         Y=scores,
    #         X=np.expand_dims(np.array([steps]), 1),
    #         opts = dict(
    #             title="Steps vs. Reward",
    #             xlabel="Episode Steps",
    #             ylabel="Episode Reward"
    #         )
    #     )
    # else:
    #     viz.scatter(Y=scores, X=steps, update="replace", win=duration_win)
    if len(mean_100) > 0:
        if not mean_score_win:
            mean_score_win =  viz.line(
                Y=mean_100,
                X=np.array(range(0, len(mean_100))),
                opts = dict(
                    title="Rolling Mean of Previous 100 Episodes",
                    xlabel="Episodes",
                    ylabel="Episode Reward"
                )
            )
        else:
            viz.line(Y=mean_100, X=np.array(range(0, len(mean_100))), update="replace", win=mean_score_win)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))
    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([torch.tensor(s) for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    s = state_batch.reshape((BATCH_SIZE, state_size))
    state_action_values = policy_net(s).gather(1, action_batch)
    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    k = non_final_next_states
    i = 0
    for s in non_final_mask:
        if (s.item() == 1):
            item = torch.tensor(tuple(k[i:i+state_size])).unsqueeze(0).to(device)
            pred = target_net(item)

            # print(pred.max(0)[0][0].detach())
            # print(pred.max(0)[1].detach())
            next_state_values[i] = pred.max(0)[0][0].detach()
        i += 1
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    losses.append(loss.item())
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        # try:
        param.grad.data.clamp_(-1, 1)
        # except Exception as e:
            # print(str(e))
    optimizer.step()



episode_steps = []
losses = []
i_episode = 0

MIN_EPSIODES = 250
MIN_SCORE = 9500

while ROLLING_MEAN_100 <= MIN_SCORE:
    i_episode +=1
    print ("starting episode: ", i_episode)
# for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = torch.FloatTensor(env.reset())
    r = 0
    steps = 0
    for t in count():
        # Select and perform an action
        action = select_action(state)
        _state, reward, done, info = env.step(action.item())
        steps += 1
        reward = torch.tensor([reward], device=device, dtype=torch.float)
        r += reward
        if not done:
            next_state = torch.tensor(_state, device=device, dtype=torch.float)
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            episode_scores.append(r)
            episode_steps.append(steps)
            print("Episode reward: ", r)
            print("Episode steps: ", steps)
            plot()
            break
    # Update the target network
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        torch.save(target_net.state_dict(),'./dqn_graph.pt')
    if i_episode > 100:
        ROLLING_MEAN_100 = np.mean(episode_scores[i_episode-100:i_episode])
        print("Rolling mean score (100): ", ROLLING_MEAN_100)
        mean_scores.append(ROLLING_MEAN_100)

print('Training Complete after ', i_episode, "episodes!")
