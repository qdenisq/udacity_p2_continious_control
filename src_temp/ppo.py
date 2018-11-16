import numpy as np
from torch.optim import Adam
import torch
from torch.nn import Module, Linear, ReLU, Tanh
from torch.distributions import Normal


class PPOAgent(Module):
    def __init__(self, **kwargs):
        super(PPOAgent, self).__init__()

        self.a_fc1 = Linear(kwargs['state_dim'], 128)
        self.a_fc2 = Linear(128, 128)
        self.mean = Linear(128, kwargs['action_dim'])
        self.log_var = Linear(128, kwargs['action_dim'])

        self.c_fc1 = Linear(kwargs['state_dim'], 128)
        self.c_fc2 = Linear(128, 128)
        self.v = Linear(128, 1)

        self.relu = ReLU()
        self.tanh = Tanh()

    def act(self, state):
        x = state
        x = self.relu(self.a_fc1(x))
        x = self.relu(self.a_fc2(x))
        mean = self.tanh(self.mean(x))
        log_var = - self.relu(self.log_var(x))

        sigmas = log_var.exp().sqrt()

        dists = Normal(mean, sigmas)
        action = dists.sample()
        action = torch.clamp(action, -1, 1)

        log_probs = dists.log_prob(action)
        log_prob = log_probs.sum(dim=-1)

        return action, log_prob

    def V(self, state):
        x = state
        x = self.relu(self.c_fc1(x))
        x = self.relu(self.c_fc2(x))
        v = self.v(x)
        return v

    def get_prob(self, state, action):
        x = state
        x = self.relu(self.a_fc1(x))
        x = self.relu(self.a_fc2(x))
        mean = self.tanh(self.mean(x))
        log_var = - self.relu(self.log_var(x))

        sigmas = log_var.exp().sqrt()

        dists = Normal(mean, sigmas)

        log_prob = dists.log_prob(action).sum(dim=-1)

        return log_prob

    def get_actor_parameters(self):
        return [*self.a_fc1.parameters(), *self.a_fc2.parameters(), *self.mean.parameters(), *self.log_var.parameters()]

    def get_critic_parameters(self):
        return [*self.c_fc1.parameters(), *self.c_fc2.parameters(), *self.v.parameters()]

class PPO:
    def __init__(self, agent=None, **kwargs):
        self.agent = agent

        self.actor_optim = Adam(agent.get_actor_parameters(), lr=kwargs['actor_lr'])
        self.critic_optim = Adam(agent.get_critic_parameters(), lr=kwargs['critic_lr'])

        self.num_epochs_actor = kwargs['num_epochs_actor']
        self.num_epochs_critic = kwargs['num_epochs_critic']

        self.discount = kwargs['discount']
        self.minibatch_size = kwargs['minibatch_size']
        self.epsilon = kwargs['epsilon']
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
            old_log_probs = []

            # Rollout
            while True:
                action, old_log_prob = self.agent.act(torch.from_numpy(state).float())
                value = self.agent.V(torch.from_numpy(state).float())
                next_state, reward, done = env.step(action.detach().numpy())

                states.append(state)
                actions.append(action.detach().numpy())
                rewards.append(reward)
                dones.append(done)
                values.append(value.detach().numpy())
                old_log_probs.append(old_log_prob.detach().numpy())

                state = next_state

                if np.any(done):
                    break

            # Calc adv

            states = np.asarray(states)
            actions = np.asarray(actions)
            rewards = np.asarray(rewards)
            dones = np.asarray(dones)
            values = np.asarray(values)
            old_log_probs = np.asarray(old_log_probs)

            T = rewards.shape[0]
            last_return = np.zeros(rewards.shape[1])
            returns = np.zeros(rewards.shape)
            advantages = np.zeros(rewards.shape)

            for t in reversed(range(T)):
                last_return = rewards[t] + last_return * self.discount * (1 - dones[t])
                returns[t] = last_return

                advantages[t] = returns[t] - values[t].squeeze()

            # Norm ?

            # Update

            actions = torch.from_numpy(actions).float().view(-1, env.get_action_dim())
            states = torch.from_numpy(states).float().view(-1, env.get_state_dim())
            returns = torch.from_numpy(returns).float().view(-1, 1)
            advantages = torch.from_numpy(advantages).float().view(-1, 1)
            old_log_probs = torch.from_numpy(old_log_probs).float().view(-1, 1)

            num_updates = actions.shape[0] // self.minibatch_size

            for k in range(self.num_epochs_actor):
                for _ in range(num_updates):
                    idx = np.random.randint(0, actions.shape[0], self.minibatch_size)
                    advantages_batch = advantages[idx]
                    returns_batch = returns[idx]
                    old_log_probs_batch = old_log_probs[idx]
                    states_batch = states[idx]
                    actions_batch = actions[idx]

                    new_log_probs = self.agent.get_prob(states_batch, actions_batch)

                    ratio = (new_log_probs - old_log_probs_batch).exp()
                    clipped = torch.clamp(ratio, 1. - self.epsilon, 1. + self.epsilon)
                    surr = torch.min(ratio, clipped) * returns_batch
                    objective = -surr.mean()

                    self.actor_optim.zero_grad()
                    objective.backward()
                    self.actor_optim.step()

            for k in range(self.num_epochs_critic):
                for _ in range(num_updates):
                    idx = np.random.randint(0, actions.shape[0], self.minibatch_size)

                    returns_batch = returns[idx]
                    states_batch = states[idx]

                    values_pred = self.agent.V(states_batch)

                    critic_loss = torch.nn.MSELoss()(values_pred, returns_batch)

                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    self.critic_optim.step()

            score = np.sum(rewards, axis=0).mean()
            print("episode: {} | score:{:.4f} | action_mean: {:.2f}, action_std: {:.2f}".format(episode, score, actions.mean(), actions.std()))

        pass