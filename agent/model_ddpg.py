import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor2(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor2, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(96*96, fc_units)
        self.fc15 = nn.Linear(fc_units, 128)
        self.fc2 = nn.Linear(128, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc15.weight.data.uniform_(*hidden_init(self.fc15))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        # print(state.shape)
        state=state.reshape(-1,96*96)
        x = F.relu(self.fc1(state)) # (64,256)
        x = F.relu(self.fc15(x)) # (64,128)
        output = abs(torch.tanh(self.fc2(x)))
        return output # ([64, 5])


class Critic2(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=256, fc2_units=128, fc3_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic2, self).__init__()
        # action_size = self.action.shape[1]
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(96*96, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        #([state]:(64, 96*96),   [action]: (64,1))
        state = state.reshape(-1,96*96) # [x, 96*96]
        # self.fcs1(state) --> [64,256] >> F.leaky_relu : [64,256]        
        xs = F.leaky_relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1) # fcs1_units+action_size ([64, 256+5])
        # hope got 256+5 = 261
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)
