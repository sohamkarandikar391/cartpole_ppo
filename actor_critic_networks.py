import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal

def get_activation(activation):
    if activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "tanh":
        return nn.Tanh()
    else:
        return nn.ReLU()

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims, activation="relu"):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        layers = []
        input_dim = self.state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(get_activation(activation))
            input_dim = hidden_dim
        
        self.network = nn.Sequential(*layers) #star before layers unpacks the list into individual arguments
        self.mean_layer = nn.Linear(input_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = self.network(state)
        mean = self.mean_layer(x)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std
        
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dims, activation="relu"):
        super(Critic, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(get_activation(activation))
            input_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self.value_layer = nn.Linear(input_dim, 1)
    
    def forward(self, state):
        x = self.network(state)
        value = self.value_layer(x)
        return value