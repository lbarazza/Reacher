from pathlib import Path
from collections import deque
from unityagents import UnityEnvironment
import numpy as np
import pandas as pd
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
run_name = 'sample_test4' # name of the current test

checkpoint_file_path = 'checkpoints/' + run_name + '.tar'
csv_file_path = 'csv/' + run_name + '.csv'
update_rate = 1

checkpoint_file = Path(checkpoint_file_path)
if checkpoint_file.is_file():
    start_episode = agent.load(checkpoint_file_path) + 1
else:
    start_episode = 0

csv_file = Path(csv_file_path)

# add headers to csv file
if not csv_file.is_file():
    data_frame=pd.DataFrame(data={
                                    'episode': [],
                                    'reward': [],
                                    'average reward': []
                                 },
                            columns = ['episode', 'reward', "average reward"])
    data_frame.to_csv(csv_file_path)

rets = deque(maxlen=100)
for episode in range(start_episode, 1000):
    ret = 0
    data = {
            'episode': [],
            'reward': [],
            'average reward': []
           }
    # reset environment
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations
    while True:
        action = agent.choose_action(state)
        env_info = env.step(action)[brain_name]
        new_state = env_info.vector_observations
        reward = env_info.rewards[0]
        done = env_info.local_done
        agent.step(state, action, reward, new_state, done)
        ret += reward
        state = new_state
        if np.any(done):
            break

    # print out results
    rets.append(ret)
    average_reward = sum(rets)/len(rets)
    print('Episode: {}\t Score: {}\t Avg. Reward: {}\t N. Steps: {}\t Std.: {}'.format(episode, ret, average_reward, agent.nSteps, agent.std))

    if average_reward >= 30:
        print("\t--> SOLVED! <--\t")
        break

    data['episode'].append(episode)
    data['reward'].append(ret)
    data['average reward'].append(average_reward)

    # save stats and checkpoint
    if episode%update_rate==0:
        agent.save(checkpoint_file_path, episode)
        data_frame = pd.DataFrame(data=data)
        data_frame.set_index('episode')
        with open(csv_file_path, 'a') as f:
            data_frame.to_csv(f, header=False)

env.close()
