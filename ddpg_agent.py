import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import normal
from networks.critic_net import Critic
from networks.actor_net import Actor
from replay_buffer import ReplayBuffer

class DDPGAgent:
    def __init__(self, nS, nA, lr_actor, lr_critic, gamma, tau, batch_size, memory_length, no_op, net_update_rate, std_initial, std_final, std_decay_frames):

        # initialize actor
        self.actor = Actor(nS, nA)
        self.actor_target = Actor(nS, nA)
        self.lr_actor = lr_actor
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        # initialize critic
        self.critic = Critic(nS, nA)
        self.critic_target = Critic(nS, nA)
        self.lr_critic = lr_critic
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # initialize memory hyperparamters
        self.memory = ReplayBuffer(memory_length)
        self.batch_size = batch_size
        self.no_op = no_op

        # initialize hyperparameters
        self.nS = nS
        self.nA = nA
        self.gamma = gamma
        self.tau = tau
        self.net_update_rate = net_update_rate

        # initialize noise
        self.std_initial = std_initial
        self.std_final = std_final
        self.std_decay_frames = std_decay_frames
        self.std_decrease = (self.std_initial-self.std_final)/self.std_decay_frames
        self.std = self.std_initial
        self.noise_distribution = normal.Normal(0, self.std)#0.15

        self.nSteps = 0

    # choose an agent
    def choose_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state).float()
            action = self.actor(state)
        action += self.noise_distribution.sample((self.nA,))
        action = np.clip(action.numpy(), -1, 1)
        return action

    # make the agent learn from its experiences
    def step(self, state, action, reward, new_state, done):
        self.nSteps += 1
        # update memory
        self.memory.add((state, action, reward, new_state, done))
        # learn - update actor and critic
        if (self.nSteps%self.net_update_rate==0) and (len(self.memory.buffer) > self.no_op):
            states, actions, rewards, new_states, dones = self.memory.sample(self.batch_size)
            self.update_critic(states, actions, rewards, new_states, dones)
            self.update_actor(states)
            # update target nets
            self.update_targets()

        # decrease noise
        self.std = max(self.std-self.std_decrease, self.std_final)
        self.noise_distribution = normal.Normal(0, self.std)

    def update_critic(self, states, actions, rewards, new_states, dones):
        # caluclate td targets with target actor and critic network
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
        # update actor_target's parameters
        for target_parameter, local_parameter in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_parameter.data.copy_((1-self.tau)*target_parameter + self.tau*local_parameter)

        # update critic_target's parameters
        for target_parameter, local_parameter in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_parameter.data.copy_((1-self.tau)*target_parameter + self.tau*local_parameter)

    # save checkpoint
    def save(self, checkpoint_path, episode):
        torch.save({
                    'actor': self.actor.state_dict(),
                    'actor_target': self.actor_target.state_dict(),
                    'actor_optimizer': self.actor_optimizer.state_dict(),
                    'critic': self.critic.state_dict(),
                    'critic_target': self.critic_target.state_dict(),
                    'critic_optimizer': self.critic_optimizer.state_dict(),
                    'std': self.std,
                    'episode': episode
        }, checkpoint_path)

    # load checkpoint
    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.std = checkpoint['std']
        self.noise_distribution = normal.Normal(0, self.std)
        return checkpoint['episode']

# TODO: implement Ornstein Uhlenbeck Noise
#
#class OUNoise:
#    def __init__(self, mu, std):
#        self.mu = mu
#        self.std = std
#
#    def OU(self):
#        pass
#
