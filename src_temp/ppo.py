import numpy as np
from torch.optim import Adam
import torch

class PPO:
    def __init__(self, agent=None, **kwargs):
        self.agent = agent

        self.actor_optim = Adam(agent.get_actor_parameters(), lr=kwargs['actor_lr'])
        self.critic_optim = Adam(agent.get_critic_parameters(), lr=kwargs['critic_lr'])

        self.num_epochs_actor = kwargs['num_epochs_actor']
        self.num_epochs_critic = kwargs['num_epochs_critic']

        self.discount = kwargs['discount']
        pass

    def train(self, env, num_episodes):
        for episode in range(num_episodes):
            state = env.reset()

            # Experiences
            states = []
            actions = []
            rewards = []
            dones = []
            values = []

            # Rollout
            while True:
                action = self.agent.act(torch.from_numpy(state).float())
                value = self.agent.V(torch.from_numpy(state).float())
                next_state, reward, done = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                values.append(value)

                state = next_state

                if np.any(done):
                    break

            # Calc adv

            states = np.asarray(states)
            actions = np.asarray(actions)
            rewards = np.asarray(rewards)
            dones = np.asarray(dones)


            T = rewards.shape[0]
            last_return = np.zeros(rewards.shape[1])
            returns = np.zeros(rewards.shape)

            for t in reversed(range(T)):
                last_return = last_return * self.discount + rewards[t]
                returns[t] = last_return



            values = self.agent.V()

            advantages = []




            # Update
            for k in range(self.num_epochs_actor):
                actor_loss = None


            for k in range(self.num_epochs_critic):
                crtitic_loss = None







        pass