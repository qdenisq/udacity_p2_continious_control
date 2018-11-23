import json
from pprint import pprint
from src.environment import ReacherEnvironment
import torch
import numpy as np


def play(*args, **kwargs):
    print(kwargs)

    fname = '../models/ppo_reacher_11_24_2018_05_09_AM.pt'
    device = 'cpu'
    kwargs['env']['seed'] = 12345
    env = ReacherEnvironment(**kwargs['env'])
    # env.reset(train_mode=False)

    agent = torch.load(fname).to(device)
    print(agent)
    agent.eval()
    for i in range(1):
        done = False
        state = env.reset(train_mode=True)
        rewards = []
        while not np.any(done):
            if 'ppo' in fname:
                # agent.eval()

                action, _, _, _ = agent(torch.from_numpy(state).float().to(device))
            elif 'ddpg' in fname:
                action = agent.act(torch.from_numpy(state).float().to(device))
            else:
                raise ValueError('Unknown agent type. Please make sure that fname has either "ppo" or "ddpg".')
            action = action.detach().cpu().numpy()
            action = np.clip(action, -1, 1)
            env.step(action)
            state, reward, done = env.step(action)
            rewards.append(reward)
            score = np.mean(np.asarray(rewards).sum(axis=0))
            print("\r play #{} | score: {}".format(i+1, score), end='')


if __name__ == '__main__':
    with open('../config.json') as data_file:
        data = json.load(data_file)

    pprint(data)
    kwargs = data
    play(**kwargs)