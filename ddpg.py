from unityagents import UnityEnvironment
import numpy as np

env = UnityEnvironment(file_name="Reacher.app", seed=42)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]

num_agents = len(env_info.agents)

action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]

env_info = env.reset(train_mode=True)[brain_name]
states = env_info.vector_observations
scores = np.zeros(num_agents)


from ddpg_agent import DDPGAgent


agent = DDPGAgent(nS=state_size,
                  nA=action_size,
                  lr_actor=0.0005,
                  lr_critic=0.0005,
                  gamma=0.99,
                  batch_size=60,#15
                  tau=0.001,
                  memory_length=int(1e6),
                  no_op=int(1e3))



for epoch in range(1000):
    ret = 0
    while True:
        #actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        #actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1

        actions = agent.choose_action(states)

        #print(actions)
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        reward = env_info.rewards[0]                         # get reward (for each agent)
        #print(rewards)
        dones = env_info.local_done                        # see if episode finished
        agent.step(states, actions, reward, next_states, dones)
        ret += reward                      # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break

    print('Episode: {}\tScore: {}\t N. Steps: {}\t Std.: {}'.format(epoch, ret, agent.nSteps, agent.std))
    if epoch%50==0:
        agent.save("checkpoints/reacher-test4.tar")
env.close()
