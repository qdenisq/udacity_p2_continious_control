# Udacity Deep Reinforcement Learning - Continuous Control (Project 2)
## Introduction

In this project, the goal is to teach an agent to move a double-jointed arm to a target 
location and keep it in the target location as long as possible. The agent has access to 
the environment's 33 dimensional state, which consists of position, rotation, velocity, 
and angular velocities of the arm. Each of these 33 dimensional states takes on continuous
 values.  From this state, the agent learns which of four actions it should take. The four
  available actions are to control the torque of the two joints of the arm and are 
  continuous values between [-1,1]. The agent gets a reward of +0.1 for each step that 
  the agent's hand is in the target location. The agent is thought to have solved or 
  learned the environment when agent gets an average score of +30 over 100 consecutive 
  episodes.

---
### Requirements:
To replicate this results of this project you need to create a conda environment using 
python 3.6 and activate it.
This can be done by following Udacity's [instructions](https://github.com/udacity/deep-reinforcement-learning#dependencies)

Next you need to download the rebuild Unity Environment. Please refer to the instructions 
in the section *Getting Started* in *Udacity DRLND GitHub*
[README](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control)

You can then explore the environment by going to the *Udacity DRLND GitHub* ([here](https://github.com/udacity/deep-reinforcement-learning)) repository and open the *Continuous_Control.ipynb* file in the *p2_continuous-control* ([here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control)).

## Training
### Twenty Agent Training
To train, in your terminal run the following:
> ```console
> (drlnd) $ python3 ./train_ppo.py
> ```
This runs a script that implements PPO. Alternatively, you may want to train your agent with DDPG. 
For this, run the following:

> ```console
> (drlnd) $ python3 ./train_ddpg.py
> ```

All hyperparameters needed for running both scripts, could be found in the `congif.json`.
You will also need to specify the location of the environment file in this config:

> ```json
> {
>    "env":
>       {
>         "env_file": "Path to environment file here"
>	    }
> }
> ```

##Testing
To play a trained agent, run the following
> ```console
> (drlnd) $ python3 ./play.py [path to a trained agent file]
> ```
Trained agents could be found in the `models` folder.