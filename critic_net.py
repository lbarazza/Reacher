import torch
import torch.nn as nn
import torch.nn.functional as F
class Critic(nn.Module):
    def __init__(self, nS, nA):
        super(Critic, self).__init__()

        self.h1 = nn.Linear(nS + nA, 200)
        self.h2 = nn.Linear(200, 300)
        self.out = nn.Linear(300, 1)

    def forward(self, s, a):
        x = torch.cat((s, a), dim=1) #?
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = self.out(x)
        return x
