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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Network(nn.Module):
    '''
    Network with one hidden layer. Only Linear layers and ReLu activation (except on output layer).
    Adam optimizer.
    MSE loss function

    '''
    def __init__(self, input_size, output_size, learning_rate):
        super().__init__()

        self.inp_layer      = nn.Linear(input_size, 64)
        self.hidden_layer1  = nn.Linear(64, 64)
        self.out_layer      = nn.Linear(64, output_size)

        self.optimizer      = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_function  = nn.MSELoss()


    def forward(self, x):
        x = F.relu(self.inp_layer(x))
        x = F.relu(self.hidden_layer1(x))
        x = self.out_layer(x)
        return x


class ExperienceBuffer:
    '''
    Replay buffer to store (state, action, reward, next state, done) experiences.
    A sampling method is provided alongside.

    '''
    def __init__(self, n_state, maxsize, batch_size, dev):
        '''

        :param n_state:     dimension of state vector
        :param maxsize:     capacity (maximum amount of experience in the buffer)
        :param batch_size:  batch size used during the sampling
        :param dev:         [cpu; gpu] device for pytorch
        '''
        self.dev            = dev
        self.capacity       = maxsize
        self.index_cntr     = 0

        self.state_buffer   = np.zeros((maxsize, n_state), dtype=np.float32)
        self.action_buffer  = np.zeros(maxsize, dtype=np.int32)
        self.reward_buffer  = np.zeros(maxsize, dtype=np.float32)
        self.n_state_buffer = np.zeros((maxsize, n_state), dtype=np.float32)
        self.done_buffer    = np.zeros(maxsize, dtype=np.int32)

        self.sample_batch_size = batch_size

    def is_full(self):
        '''

        :return: True if buffer is full, False if not
        '''
        if self.index_cntr >= self.capacity:
            return True
        else:
            return False

    def add(self, state, action, reward, n_state, done):
        ''' Adding a new experience to the buffer.

        :param state:
        :param action:
        :param reward:
        :param n_state:
        :param done:
        :return:
        '''
        i = self.index_cntr%self.capacity

        self.state_buffer[i]    = state
        self.action_buffer[i]   = action
        self.reward_buffer[i]   = reward
        self.n_state_buffer[i]  = n_state
        self.done_buffer[i]     = done

        self.index_cntr += 1

    def sample_batch(self):
        ''' Sampling an experience batch from the replay buffer.

        :return:    s   : torch tensor containing the sampled state vectors
                    a   : numpy array containing the sampled action indices
                    r   : torch tensor containing the sampled reward signals
                    n_s : torch tensor containing the sampled next state vectors
                    d   : torch tensor containing the sampled done flags
        '''
        batch = np.random.choice(self.capacity, self.sample_batch_size, replace=False)

        s   = torch.tensor(self.state_buffer[batch]).to(self.dev)
        a   = self.action_buffer[batch]
        r   = torch.tensor(self.reward_buffer[batch]).to(self.dev)
        n_s = torch.tensor(self.n_state_buffer[batch]).to(self.dev)
        d   = torch.tensor(self.done_buffer[batch]).to(self.dev)

        return s,a,r,n_s,d


class Agent:
    ''' Implements an agent utilizing a DQN algorithm to train it with replay buffer and fixed target.

    '''
    def __init__(self, environment, discount_factor, exp_buffer_size, batch_size, target_network_update_freq, learning_rate):
        '''

        :param environment:                 opengym environment
        :param discount_factor:             [gamma] discount factor
        :param exp_buffer_size:             maximum capacity of the replay buffer
        :param batch_size:                  sampled batch size
        :param target_network_update_freq:  number of steps the fixed network is updated
        :param learning_rate:               learning rate
        '''
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actions = environment.action_space
        self.state_dims = len(environment.observation_space.high)
        self.gamma = discount_factor
        self.batch_size = batch_size
        self.batch_i = np.arange(self.batch_size, dtype=np.int32)
        self.experience_buffer = ExperienceBuffer(self.state_dims, exp_buffer_size, self.batch_size, self.dev)
        self.C = target_network_update_freq
        self.Ci = 0
        self.network = Network(self.state_dims, self.actions.n, learning_rate).to(self.dev)
        self.fixed_network = Network(self.state_dims, self.actions.n, learning_rate).to(self.dev)
        self.__sync_target()


    def __sync_target(self):
        ''' Copying weights from network to target network (fixing current network)

        :return:
        '''
        self.fixed_network.load_state_dict(self.network.state_dict())
        self.fixed_network.eval()

    def __handle_fixed_target(self):
        ''' Incrementing counter for the fixed target network and updates in case it reaches the pre-set value.

        :return:
        '''
        self.Ci += 1
        if self.Ci % self.C == 0:
            self.__sync_target()

    def __update(self):
        ''' Update cycle of the network based on the DQN algorithm.

        :return:
        '''
        s,a,r,n_s,d = self.experience_buffer.sample_batch()

        self.network.optimizer.zero_grad()

        target      = r + ((1-d) * self.gamma * (torch.max(self.fixed_network(n_s), dim=1)[0]))
        estimate    = self.network(s)[self.batch_i,a]

        loss = self.network.loss_function(target, estimate).to(self.dev)
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(self.network.parameters(), 2.0)
        self.network.optimizer.step()

        self.__handle_fixed_target()

        return loss.item()

    def add_experience(self, state, action, reward, next_state, done):
        ''' Load one experience to the agent's replay buffer. This method calls the update function as well.

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return: Returns the loss once the experience buffer is full and learning can be started.
        '''
        self.experience_buffer.add(state, action, reward, next_state, done)
        if self.experience_buffer.is_full():
            loss = self.__update()
            return loss
        else:
            return 0


    def action(self, state, eps_k):
        ''' Epsilon-greedy action strategy.

        :param state:
        :param eps_k:
        :return:
        '''
        if np.random.random() <= eps_k:
            return self.actions.sample()
        else:
            s = torch.tensor([state]).to(self.dev)
            q_values = self.network(s)
            return torch.argmax(q_values).item()
