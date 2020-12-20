#################################################
#                                               #
# EL2805 Reinforcement Learning                 #
# Computer Lab 2                                #
# Problem 1                                     #
#                                               #
# Author: Tamas Faitli (19960205-T410)          #
# Description: This module implements a DQN     #
#              network.                         #
#                                               #
#################################################

import numpy as np
# from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.inp_layer      = nn.Linear(input_size, 128)
        self.hidden_layer   = nn.Linear(128, 64)
        self.out_layer      = nn.Linear(64, output_size)


    def forward(self, x):
        x = F.relu(self.inp_layer(x))
        x = F.relu(self.hidden_layer(x))
        x = self.out_layer(x)
        return x


class ExperienceBuffer:
    def __init__(self, maxsize, batch_size):
        self.maxsize = maxsize
        self.buffer = []
        self.sample_batch_size = batch_size

    def is_full(self):
        if len(self.buffer) >= self.maxsize:
            return True
        else:
            return False

    def add(self, experience):
        if self.is_full():
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample_batch(self):
        return random.sample(self.buffer, self.sample_batch_size)



class Agent:
    def __init__(self, environment, discount_factor, exp_buffer_size, batch_size, target_network_update_freq, learning_rate):
        self.actions = environment.action_space
        self.state_dims = len(environment.observation_space.high)
        self.gamma = discount_factor
        self.batch_size = batch_size
        self.experience_buffer = ExperienceBuffer(exp_buffer_size, self.batch_size)
        self.C = target_network_update_freq
        self.Ci = 0
        self.network = Network(self.state_dims, self.actions.n)
        self.fixed_network = Network(self.state_dims, self.actions.n)
        self.__sync_target()
        # self.optimizer = optim.SGD(self.network.parameters(), lr=learning_rate)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __sync_target(self):
        self.fixed_network.load_state_dict(self.network.state_dict())

    def __handle_fixed_target(self):
        self.Ci += 1
        if self.Ci >= self.C:
            self.__sync_target()
            self.Ci = 0

    def __update(self):
        batch = self.experience_buffer.sample_batch()

        y = torch.zeros(1,1,self.batch_size, device=self.dev ,requires_grad=False)
        target = torch.zeros(1,1,self.batch_size, device=self.dev, requires_grad=False)

        self.optimizer.zero_grad()

        i = 0
        # b[0]: state, b[1]: action, b[2]: reward, b[3]: next_state, b[4]: done or not
        for b in batch:
            y[:,:,i] = b[2]
            if b[4]:
                y[:,:,i] += self.gamma * torch.max(self.fixed_network(b[3]))
            target[:,:,i] = self.network(b[0])[b[1]]
            i += 1

        loss = F.mse_loss(y, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 2.0)
        self.optimizer.step()


        self.__handle_fixed_target()

    def add_experience(self, state, action, reward, next_state, done):
        exp = (torch.from_numpy(state).float(), action, reward,
               torch.from_numpy(next_state).float(), done)
        self.experience_buffer.add(exp)
        if self.experience_buffer.is_full():
            self.__update()

    def action(self, state, eps_k):
        if random.random() < eps_k:
            return self.actions.sample()
        else:
            q_values = self.network(torch.from_numpy(state).float())
            return torch.argmax(q_values).item()
