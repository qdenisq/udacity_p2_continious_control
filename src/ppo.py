import numpy as np
import progressbar as pb
import torch
from torch.nn import MSELoss
import time

class PPO:
    def __init__(self, *args, agent=None, **kwargs):
        self.agent = agent
        self.optim = torch.optim.Adam(self.agent.get_parameters(), lr=kwargs['learning_rate'])
        self.__discount = kwargs['discount']
        self.__lambda = kwargs['lambda']
        self.__epsilon = kwargs['epsilon']
        self.__num_rollouts = kwargs['num_rollouts_per_update']
        self.__num_updates = kwargs['num_updates']
        self.__minibatch_size = kwargs['minibatch_size']
        return

    def roll_out(self, env, num_roll_outs):
        num_agents = env.get_num_agents()
        if num_roll_outs % num_agents != 0:
            raise ValueError("Number of rollouts should be dividable by number of agents: rollouts={}, agents={}"
                             .format(num_roll_outs, num_agents))
        num_episodes = num_roll_outs // num_agents
        T = env.get_episode_len()
        states = np.zeros((T, num_roll_outs, env.get_state_dim()))
        actions = np.zeros((T, num_roll_outs, env.get_action_dim()))
        rewards = np.zeros((T, num_roll_outs))
        old_probs = np.zeros((T, num_roll_outs))
        values = np.zeros((T, num_roll_outs))
        for i in range(num_episodes):
            state = env.reset(train_mode=True)
            # roll out
            j = 0
            while True:
                action, probs, v = self.agent.act(torch.Tensor(state))
                next_state, reward, done = env.step(action.squeeze())

                rewards[j, num_agents*i: num_agents*(i+1)] = reward
                old_probs[j, num_agents*i: num_agents*(i+1)] = probs
                values[j, num_agents*i: num_agents*(i+1)] = v.squeeze()
                states[j, num_agents*i: num_agents*(i+1), :] = state
                actions[j, num_agents*i: num_agents*(i+1), :] = action

                state = next_state
                j += 1
                if np.any(done):
                    break
        return states, actions, rewards, old_probs, values

    def compute_loss(self, new_probs, old_probs, v_pred, returns, advantages, mus, sigmas):
        # actor loss
        ratio = (new_probs - old_probs).exp()
        clip = torch.clamp(ratio, 1. - self.__epsilon, 1. + self.__epsilon)
        clipped_surrogate = torch.min(ratio, clip) * returns
        surrogate = -torch.mean(clipped_surrogate)
        # critic loss
        critic_loss = (v_pred - returns.reshape(-1, 1)).pow(2).mean()
        # entropy loss
        ent_loss = 0.

        #action penalty
        action_penalty = 1e3 * torch.abs(torch.mean(mus)) + 1e3 * torch.mean(sigmas)

        loss = surrogate + critic_loss + ent_loss
        return loss

    def train(self, env, num_steps):
        mean_score = []
        for step in range(num_steps):
            states, actions, rewards, old_probs, values = self.roll_out(env, self.__num_rollouts)

            T = rewards.shape[0]
            values = np.append(values, np.zeros((1, rewards.shape[1])), axis=0)
            last_advantage = np.zeros(rewards.shape[1])
            last_return = np.zeros(rewards.shape[1])
            advantages = np.zeros(rewards.shape)
            returns = np.zeros(rewards.shape)
            for t in reversed(range(T)):
                delta = rewards[t] + self.__discount * values[t+1] - values[t]
                last_advantage = delta + self.__discount * self.__lambda * last_advantage
                advantages[t] = last_advantage
                last_return = rewards[t] + self.__discount * last_return
                returns[t] = last_return

            # normalize advanatge and returns
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
            returns = (returns - returns.mean()) / (returns.std() + 1e-10)


            #
            # discount = self.__discount ** np.arange(rewards.shape[0])
            # rewards_discounted = rewards * discount[:, np.newaxis]
            # returns = rewards_discounted[::-1].cumsum(axis=0)[::-1]
            # mean_rewards = np.mean(returns, axis=1)
            # std_rewards = np.std(returns, axis=1) + 1.0e-10
            # returns = (returns - mean_rewards[:, np.newaxis]) / std_rewards[:, np.newaxis]

            # returns = (returns - returns.mean()) / (returns.std() + 1e-5)
            #
            #
            #
            # #
            # # v = np.zeros(rewards.shape)
            # # prev_rets = np.zeros(rewards.shape[1])
            # # for i in reversed(range(rewards.shape[0] - 1)):
            # #     v[i] = rewards[i] + self.__discount * prev_rets
            # #     prev_rets = v[i]
            # # rewards_future = v
            #
            # mean_rewards = np.mean(rewards_future, axis=1)
            # std_rewards = np.std(rewards_future, axis=1) + 1.0e-10
            # rewards_normalized = (rewards_future - mean_rewards[:, np.newaxis]) / std_rewards[:, np.newaxis]

            returns = torch.tensor(returns.flatten(), dtype=torch.float)
            advantages = torch.tensor(advantages.flatten(), dtype=torch.float)
            states = torch.tensor(states.reshape(-1, env.get_state_dim()), dtype=torch.float)
            actions = torch.tensor(actions.reshape(-1, env.get_action_dim()), dtype=torch.float)
            old_probs = torch.tensor(old_probs.flatten(), dtype=torch.float)
            # old_probs = torch.tensor(old_probs.flatten(), dtype=torch.float, requires_grad=False)

            for i in range(self.__num_updates):

                idx = np.random.randint(0, returns.shape[0], self.__minibatch_size)

                returns_batch = returns[idx]
                advantages_batch = advantages[idx]
                states_batch = states[idx]
                actions_batch = actions[idx]
                old_probs_batch = old_probs[idx]



                # calc new probs
                new_probs, v_pred, mus, sigmas = self.agent.get_prob_and_v(states_batch, actions_batch)
                loss = self.compute_loss(new_probs, old_probs_batch, v_pred, returns_batch, advantages_batch, mus, sigmas)
                self.optim.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1)
                self.optim.step()

            self.__epsilon *= .999

            mean_score.append(np.mean(np.sum(rewards, axis=0)))

            # display some progress every 20 iterations
            if (step + 1) % 20 == 0:
                print("Episode: {0:d}, score: {1:f}".format(step + 1, mean_score[-1]))
            print("Episode: {0:d}, score: {1:f}".format(step + 1, mean_score[-1]))
            print(self.agent.log_var)



    def save(self, fname):
        pass

    def load(self, fname):
        pass

