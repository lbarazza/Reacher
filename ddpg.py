from unityagents import UnityEnvironment
import numpy as np

env = UnityEnvironment(file_name="Reacher.app")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]

num_agents = len(env_info.agents)

action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]

env_info = env.reset(train_mode=True)[brain_name]     # reset the environment
states = env_info.vector_observations                  # get the current state (for each agent)
scores = np.zeros(num_agents)                          # initialize the score (for each agent)


from ddpg_agent import DDPGAgent
agent = DDPGAgent(nS=state_size,
                  nA=action_size,
                  gamma=0.99,
                  batch_size=5,
                  tau=0.001,
                  memory_length=int(1e6),
                  no_op=int(1e3))




while True:
    #actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
    #actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1

    actions = agent.choose_action(states)


    env_info = env.step(actions)[brain_name]           # send all actions to tne environment
    next_states = env_info.vector_observations         # get next state (for each agent)
    rewards = env_info.rewards                         # get reward (for each agent)
    dones = env_info.local_done                        # see if episode finished
    agent.step(states, actions, rewards, next_states, dones)
    scores += env_info.rewards                         # update the score (for each agent)
    states = next_states                               # roll over states to next time step
    if np.any(dones):                                  # exit loop if episode finished
        break
print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
env.close()
