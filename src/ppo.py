import numpy as np
import progressbar as pb
import torch
import time

class PPO:
    def __init__(self, *args, agent=None, **kwargs):
        self.agent = agent
        self.optim = torch.optim.Adam(self.agent.parameters(), lr=kwargs['learning_rate'])
        self.__discount = kwargs['discount']
        self.__epsilon = kwargs['epsilon']
        self.__num_rollouts = kwargs['num_rollouts_per_update']
        self.__num_updates = kwargs['num_updates']
        return

    def surrogate(self, smth):
        pass

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
        for i in range(num_episodes):
            state = env.reset(train_mode=True)
            # roll out
            j = 0
            while True:
                action, probs = self.agent.act(torch.Tensor(state))
                next_state, reward, done = env.step(action.squeeze())

                rewards[j, 20*i:20*(i+1)] = reward
                old_probs[j, 20*i:20*(i+1)] = probs
                states[j, 20*i:20*(i+1), :] = state
                actions[j, 20*i:20*(i+1), :] = action

                state = next_state
                j += 1
                if np.any(done):
                    break
        return states, actions, rewards, old_probs

    def train(self, env, num_steps):
        mean_score = []
        widget = ['training loop: ', pb.Percentage(), ' ',
                  pb.Bar(), ' ', pb.ETA()]
        timer = pb.ProgressBar(widgets=widget, maxval=num_steps).start()
        for step in range(num_steps):
            t0 = time.time()

            states, actions, rewards, old_probs = self.roll_out(env, self.__num_rollouts)

            t1 = time.time()

            discount = self.__discount ** np.arange(rewards.shape[0])
            rewards_discounted = rewards * discount[:, np.newaxis]
            rewards_future = rewards_discounted[::-1].cumsum(axis=0)[::-1]
            mean_rewards = np.mean(rewards_future, axis=1)
            std_rewards = np.std(rewards_future, axis=1) + 1.0e-10
            rewards_normalized = (rewards_future - mean_rewards[:, np.newaxis]) / std_rewards[:, np.newaxis]

            rewards_normalized = torch.tensor(rewards_normalized.flatten(), dtype=torch.float, requires_grad=False)
            states = torch.tensor(states.reshape(-1, env.get_state_dim()), dtype=torch.float, requires_grad=False)
            actions = torch.tensor(actions.reshape(-1, env.get_action_dim()), dtype=torch.float, requires_grad=False)
            old_probs = torch.tensor(old_probs.flatten(), dtype=torch.float, requires_grad=False)

            t2 = time.time()
            get_prob_time = 0.
            t5 = 0.
            t6 = 0.
            t7 = 0.
            for i in range(self.__num_updates):
                # calc new probs
                t4 = time.time()
                new_probs = self.agent.get_prob(states, actions)
                get_prob_time += time.time() - t4

                t = time.time()
                ratio = new_probs / old_probs
                clip = torch.clamp(ratio, 1 - self.__epsilon, 1 + self.__epsilon)
                clipped_surrogate = torch.min(ratio * rewards_normalized, clip * rewards_normalized)
                surrogate = torch.mean(clipped_surrogate)
                # surrogate = torch.mean(rewards_normalized)

                t5 += time.time() - t
                t = time.time()
                self.optim.zero_grad()
                surrogate.backward()
                t6 += time.time() - t

                t = time.time()
                self.optim.step()
                t7 += time.time() - t
            t3 = time.time()
            # print("rollout: {}, processing: {}, update: {}, get_prob: {}, {} backprop: {} {}".format(t1-t0, t2-t1, t3-t2, get_prob_time, t5, t6, t7))

            self.__epsilon *= .999

            mean_score.append(np.mean(np.sum(rewards, axis=0)))

            # display some progress every 20 iterations
            if (step + 1) % 20 == 0:
                print("Episode: {0:d}, score: {1:f}".format(step + 1, mean_score[-1]))
            print("Episode: {0:d}, score: {1:f}".format(step + 1, mean_score[-1]))

            # update progress widget bar
            timer.update(step + 1)
        timer.finish()

    def save(self, fname):
        pass

    def load(self, fname):
        pass

