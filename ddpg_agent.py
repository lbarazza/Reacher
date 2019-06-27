import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import normal
from networks.critic_net import Critic
from networks.actor_net import Actor
from replay_buffer import ReplayBuffer

class DDPGAgent:
    def __init__(self, nS, nA, lr_actor, lr_critic, gamma, tau, batch_size, memory_length, no_op):
        self.actor = Actor(nS, nA)
        self.actor_target = Actor(nS, nA)
        self.lr_actor = lr_actor
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(nS, nA)
        self.critic_target = Critic(nS, nA)
        self.lr_critic = lr_critic
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

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

        self.net_update_rate = 1

        # initialize noise
        self.std_initial = 0.15
        self.std_final = 0.025
        self.std_decay_frames = 200000
        self.std_decrease = (self.std_initial-self.std_final)/self.std_decay_frames
        self.std = self.std_initial
        self.noise_distribution = normal.Normal(0, self.std)#0.15

        self.nSteps = 0

    def choose_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state).float() #.unsqueeze(0)
            action = self.actor(state)
        action += self.noise_distribution.sample((self.nA,))
        action = np.clip(action.numpy(), -1, 1)
        return action

        ## add noise <----



    def step(self, state, action, reward, new_state, done):
        self.nSteps += 1
        self.memory.add((state, action, reward, new_state, done))
        if (self.nSteps%self.net_update_rate==0) and (len(self.memory.buffer) > self.no_op):
            states, actions, rewards, new_states, dones = self.memory.sample(self.batch_size)
            self.update_critic(states, actions, rewards, new_states, dones)
            self.update_actor(states)#, actions, rewards, new_states, dones)
            self.update_targets()

        # decrease noise
        self.std = max(self.std-self.std_decrease, self.std_final)
        self.noise_distribution = self.noise_distribution = normal.Normal(0, self.std)
        #print(self.std)

    def update_critic(self, states, actions, rewards, new_states, dones):
        # caluclate td targets with targer actor and critic network
        with torch.no_grad():
            maximizing_actions = self.actor_target(new_states)
            y = rewards + self.gamma * (1-dones) * self.critic_target(new_states, maximizing_actions)
        critic_predicition = self.critic(states, actions)

        # caluclate loss and perform backpropagation
        loss = F.mse_loss(critic_predicition, y)
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        # apply the gradients to the network parameters
        self.critic_optimizer.step()

    def update_actor(self, states): # ??? ->> states, actions, rewards, new_states, dones
        maximizing_actions = self.actor(states)
        objective_func = -torch.mean(self.critic(states, maximizing_actions))
        self.actor_optimizer.zero_grad()
        objective_func.backward()
        self.actor_optimizer.step()

    def update_targets(self):

        # introduce update rate?

        # update actor_target's parameters
        for target_parameter, local_parameter in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_parameter.data.copy_((1-self.tau)*target_parameter + self.tau*local_parameter)

        # update critic_target's parameters
        for target_parameter, local_parameter in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_parameter.data.copy_((1-self.tau)*target_parameter + self.tau*local_parameter)

    def save(self, checkpoint_path):
        torch.save({
                    'actor': self.actor.state_dict(),
                    'actor_target': self.actor_target.state_dict(),
                    'critic': self.critic.state_dict(),
                    'critic_target': self.critic_target.state_dict()
        }, checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)


#
#class noise:
#    def __init__(self, mu, std):
#        self.mu = mu
#        self.std = std
#
#    def normal(self):
#        return normal.Normal(self.mu, self.std)
#
#    # TODO: implement Ornstein Uhlenbeck Noise
#    def OU(self):
#        pass
#


















#
