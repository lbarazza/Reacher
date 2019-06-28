import pandas as pd
import matplotlib.pyplot as plt

# read the csv file created

data_frame = pd.read_csv('csv/reacher-test6.csv')

episodes = data_frame['episode']
rewards = data_frame['reward']
average_rewards = data_frame['average reward']

# plot agent's progress over time
plt.plot(episodes, rewards, color='dodgerblue')
plt.plot(episodes, average_rewards, color='blue')
plt.axhline(y=30, color='red')

plt.show()
