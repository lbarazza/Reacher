# Report

## Deep Deterministic Policy Gradients
The Deep Deterministic Policy Gradient (DDPG) tries to solve some of the major limitations of the famous 
[DQN algorithm](https://github.com/lbarazza/Banana-Collector)

## Networks
Two separate networks for actor and critic networks have been used. Both networks have two hidden layers with 100 neurons each 
and ReLU activation. The actor network has a tanh activation function applied to the output. Instead, the critic network doesn't 
have any.

## Hyperparameters
An informal search over the hyperparameters has been conducted with the following results:

|     Hyperparamter                          |      Value                      |
|--------------------------------------------|:-------------------------------:|
|    actor learning rate                     |          0.0005                 |
|    critic learning rate                    |          0.0005                 |
|    noise standard deviation start          |          0.15                   |
|    noise standard deviation end            |          0.025                  |
|    noise standard deviation decay frames * |          0.025                  |
|    gamma                                   |          0.99                   |
|    networks update rate                    |          1                      |
|    tau                                     |          0.001                  |
|    memory length                           |          1,000,000              |
|    replay memory start                     |          1,000                  |
|    n. no op at beginning of training       |          1,000                  |
|    batch_size                              |          60                     |
|    replay_start_size                       |          1,000                  |

* the standard deviation has been decreased linearly in the amount of frames specified

## Results
The agent was able to solve the environment in 100 episodes:
![alt text](https://raw.githubusercontent.com/lbarazza/Reacher/master/images/stats.png "DDPG stats")
(the average reward is in dark blue, while the actual single rewards are plotted in light blue)

## Improvements
The agent could be improved to:
- use OU noise instead of normal gaussian noise with zero mean
- be moved to a distributed version of DDPG such as D4PG (of course by adding more agents)
- make use of Prioritized Experience Replay
