import json
from pprint import pprint
import numpy as np
from src.environment import ReacherEnvironment
from src_temp.ppo import PPO, PPOAgent
import torch
import numpy as np
import random

def train(*args, **kwargs):
    print(kwargs)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    kwargs['ppo']['device'] = device

    torch.manual_seed(kwargs['agent']['seed'])
    np.random.seed(kwargs['agent']['seed'])
    random.seed(kwargs['agent']['seed'])

    env = ReacherEnvironment(**kwargs['env'])
    env.reset(train_mode=True)

    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    kwargs['agent']['state_dim'] = state_dim
    kwargs['agent']['action_dim'] = action_dim

    agent = PPOAgent(**kwargs['agent']).to(device)
    alg = PPO(agent=agent, **kwargs['ppo'])
    alg.train(env, 800)


if __name__ == '__main__':
    with open('../config.json') as data_file:
        data = json.load(data_file)

    pprint(data)
    kwargs = data

    train(**kwargs)