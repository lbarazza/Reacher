from collections import deque
import random
import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, length):
        # initialize the buffer
        self.length = length
        self.buffer = deque(maxlen=length)

    # add experience to the the replay sbuffer
    def add(self, x):
        self.buffer.append(x)

    # sample from the buffer
    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        return ReplayBuffer.preprocess_experiences(experiences)

    # extract all states, actions, rewards and new states from tuple and return a separate tensor for each
    @staticmethod
    def preprocess_experiences(experiences):
        states = torch.tensor(np.vstack([i[0] for i in experiences])).float()
        actions = torch.tensor(np.vstack([i[1] for i in experiences])).float()
        rewards = torch.tensor(np.vstack([i[2] for i in experiences])).float()
        new_states = torch.tensor(np.vstack([i[3] for i in experiences])).float()
        dones = torch.tensor(np.vstack([i[4] for i in experiences]).astype(np.uint8)).float()
        return states, actions, rewards, new_states, dones
