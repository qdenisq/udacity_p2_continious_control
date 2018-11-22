import json
from pprint import pprint
from src.environment import ReacherEnvironment
import torch
import numpy as np


def play(*args, **kwargs):
    print(kwargs)

    fname = '../models/ppo_reacher_11_22_2018_06_38_PM.pt'
    device = 'cpu'

    env = ReacherEnvironment(**kwargs['env'])
    env.reset(train_mode=True)

    agent = torch.load(fname).to(device)

    for i in range(2):
        done = False
        score = 0
        state = env.reset(train_mode=False)
        while not np.any(done):
            if 'ppo' in fname:
                action, _, _, _ = agent(torch.from_numpy(state).float().to(device))
            elif 'ddpg' in fname:
                action = agent.act(torch.from_numpy(state).float().to(device))
            else:
                raise ValueError('Unknown agent type. Please make sure that fname has either "ppo" or "ddpg".')
            action = action.detach().cpu().numpy()
            action = np.clip(action, -1, 1)
            env.step(action)
            state, reward, done = env.step(action)
            score += np.mean(reward)
            print("\r play #{} | score: {}".format(i+1, score), end='')


if __name__ == '__main__':
    with open('../config.json') as data_file:
        data = json.load(data_file)

    pprint(data)
    kwargs = data
    play(**kwargs)