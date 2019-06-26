import torch
import numpy as np
from networks.critic_net import Critic
from networks.actor_net import Actor
from replay_buffer import ReplayBuffer

class DDPGAgent:
    def __init__(self, nS, nA, gamma, tau, batch_size, memory_length, no_op):
        self.actor = Actor(nS, nA)
        self.actor_target = Actor(nS, nA)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(nS, nA)
        self.critic_target = Critic(nS, nA)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        # set target nets to be equal to the local net??

        # initialize memory hyperparamters
        self.memory = ReplayBuffer(memory_length)
        self.batch_size = batch_size
        self.no_op = no_op

        # initialize hyperparameters
        self.nS = nS
        self.nA = nA
        self.gamma = gamma
        self.tau = tau


    def choose_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state).float() #.unsqueeze(0)
            action = self.actor(state)
        return action.numpy()

        ## add noise <----



    def step(self, state, action, reward, new_state, done):
        self.memory.add((state, action, reward, new_state, done))
        if len(self.memory.buffer) > self.no_op:
            states, actions, rewards, new_states, dones = self.memory.sample(self.batch_size)
            self.update_critic(states, actions, rewards, new_states, dones)
            self.update_actor(states, actions, rewards, new_states, dones)
            self.update_targets()

    def update_critic(states, actions, rewards, new_states, dones):
        # caluclate td targets with targer actor and critic network
        with torch.no_grad():
            maximizing_actions = self.actor_target(new_states)
            y = rewards + self.gamma * (1-dones) * self.critic_target(new_states, maximizing_actions)
        critic_predicition = self.critic(states, actions)

        # caluclate loss and perform backpropagation
        loss = F.mse_loss(critic_predicition, y)
        self.critic_optimizer.zero_grad()
        loss.backward()
        # apply the gradients to the network parameters
        self.critic_optimizer.step()

    def update_actor(states): # ??? ->> states, actions, rewards, new_states, dones
        maximizing_actions = self.actor(states)
        objective_func = -self.critic(states, maximizing_actions)
        self.actor_optimizer.zero_grad()
        objective_func.backward()
        self.actor_optimizer.step()

    def update_targets():

        # introduce update rate?

        # update actor_target's parameters
        for target_parameter, local_parameter in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_parameter.data.copy_((1-self.tau)*target_parameter + self.tau*local_parameter)

        # update critic_target's parameters
        for target_parameter, local_parameter in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_parameter.data.copy_((1-self.tau)*target_parameter + self.tau*local_parameter)


























#
