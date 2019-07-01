import torch
import torch.nn as nn
import torch.nn.functional as F
class Actor(nn.Module):
    def __init__(self, nS, nA): # nS: state space size, nA: action space size
        super(Actor, self).__init__()

        self.h1 = nn.Linear(nS, 100)
        self.h2 = nn.Linear(100, 100)
        self.out = nn.Linear(100, nA)

    def forward(self, x):
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.tanh(self.out(x))
        return x
