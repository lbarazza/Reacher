from pathlib import Path
from collections import deque
from unityagents import UnityEnvironment
import numpy as np
from ddpg_agent import DDPGAgent

# initialize environment
env = UnityEnvironment(file_name="Reacher.app")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
state = env_info.vector_observations
state_size = state.shape[1]

# create agent
agent = DDPGAgent(nS=state_size,
                  nA=action_size,
                  lr_actor=0.0005,
                  lr_critic=0.0005,
                  gamma=0.99,
                  batch_size=60,
                  tau=0.001,
                  memory_length=int(1e6),
                  no_op=int(1e3),
                  net_update_rate=1,
                  std_initial=0.15,
                  std_final=0.025,
                  std_decay_frames=200000)

# setup csv and checkpoint files
run_name = 'sample_test2' # name of the current test

checkpoint_file_path = 'checkpoints/' + run_name + '.tar'

checkpoint_file = Path(checkpoint_file_path)
if checkpoint_file.is_file():
    agent.load(checkpoint_file_path)

rets = deque(maxlen=100)
for episode in range(0, 10):
    ret = 0

    # reset environment
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations
    while True:
        action = agent.choose_action(state)
        env_info = env.step(action)[brain_name]
        new_state = env_info.vector_observations
        reward = env_info.rewards[0]
        done = env_info.local_done
        ret += reward
        state = new_state
        if np.any(done):
            break

    # print out results
    rets.append(ret)
    average_reward = sum(rets)/len(rets)
    print('Episode: {}\t Score: {}\t Avg. Reward: {}\t N. Steps: {}\t Std.: {}'.format(episode, ret, average_reward, agent.nSteps, agent.std))

env.close()
