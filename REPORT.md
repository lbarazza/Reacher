# Report

## Deep Deterministic Policy Gradients
The Deep Deterministic Policy Gradient (DDPG) tries to solve one of the major limitations of the famous [DQN algorithm](https://github.com/lbarazza/Banana-Collector), that is the application of the algorithm in contnous action spaces.

To understand how it does this it's good to first look at the main formula for DQN:
![alt text](https://cdn-images-1.medium.com/max/800/1*KsQ46R8zyTQlKGv91xi6ww.png "DQN formula")
<sub>Taken from (www.freecodecamp.org)</sub>

As we can see from the formula, DQN is based off of maximizing the Q value, which is fine as long as we have a discrete amount of actions to choose from, but in the moment where we have an infinite (continous) amount of actions this becomes impossible.
How do we deal with this then? One solution is to create a new neural network, the actor, that apporximates the function that maps states to the actions that maximize the Q function. This way instead of using argmax over the set of q values of each action, we just plug in the state into the actor and use the action it spits out instead. The rest of the algorithm is pretty much identical to DQN, the only thing left to understand is how do we make the actor network learn? I mean, we have the TD Targets for the critic (value) network, but how do we understand which way the actor's paramters should be updated? The answer is to just follow the gradient of the Q function. Why? Well, to find the Q value of a state-action pair we plug in the state into the actor net and then use the output as the input (together with the state) of the critic (we are basically just concatenating the two neural networks). We are also using gradient ascent to find the maximum of the Q function, so the only thing we should need is the gradient of the Q function with respect to the actor's network parameters (not the critic!). So, what we do is we first use the actor network to find the maximizing action, we then use the output of that to calculate the Q value and finally we backpropagate the gradients from this Q value, but we only use the gradient we got for the parameters of the first (the actor) network and use gradient ascent on them.
Et voilà, we modified the famous DQN algorithm to work with continous action spaces.

Little trivia, DDPG was first proposed as an actor-critic network, but it's commonly also seen as a value-based method, it depends on who you ask!

## Networks
Two separate networks for actor and critic networks have been used. Both networks have two hidden layers with 100 neurons each and ReLU activation. The actor network has a tanh activation function applied to the output. Instead, the critic network doesn't have any.

## Hyperparameters
An informal search over the hyperparameters has been conducted with the following results:

|     Hyperparamter                          |      Value                      |
|--------------------------------------------|:-------------------------------:|
|    actor learning rate                     |          0.0005                 |
|    critic learning rate                    |          0.0005                 |
|    noise standard deviation start          |          0.15                   |
|    noise standard deviation end            |          0.025                  |
|    noise standard deviation decay frames * |          200000                 |
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
