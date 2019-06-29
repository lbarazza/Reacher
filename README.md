# Reacher
![alt text](https://raw.githubusercontent.com/lbarazza/Reacher/master/images/reacher_video.gif "Reacher gif")

## Project Details
This project solves the environment provided in the Udacity Reinforcement Learning Nanodegree Program. The system is a simulated robotic arm whose goal is to reach the green spot with its arm. For every time step that the arm is in the right place, a reward of +0.1 is given, otherwise it's always zero. The environment is considered solved whenever the agent reaches an average reward of +30 over 100 episodes. The state space is made up of 33 dimensions consisting of position, rotation, velocity and angular velocity for every degree of freedom of the arm. the action space is continous and 4-dimensional, every action consisiting of the torques to apply to each joint (actions are between -1 and 1).

## Dependencies
This project is implemented using Python 3.6, PyTorch, NumPy and the UnityEnvironment package. For the plotting part of the project Matplotlib is used, while for the part dealing with csv, Pandas is used.

## Installation
Download the repository with

```
git clone https://github.com/lbarazza/Reacher
```

To install the environment, download the version corresponding to your operating system:

[Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)

[Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)

[Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)

[Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

Unzip the downloaded file and place it in the root folder of the project.
Now move into the project folder and create a virtual environement where we can install all of the dependencies

```
cd Reacher
conda create --name reacher-project python=3.6 matplotlib pandas
```

Activate the virtual environment:

(Mac and Linux)
```
source activate reacher-project
```

(Windows)
```
activate reacher-project
```

Now install all the dependencies:

```
pip install -r requirements.txt
```

## Run the Code
To create a new agent and make it learn from scratch run (or resume training):

```
python ddpg.py
```
