import json
from pprint import pprint
import numpy as np
from src.environment import ReacherEnvironment
from src_temp.ppo import PPO, PPOAgent

def train(*args, **kwargs):
    print(kwargs)

    env = ReacherEnvironment(**kwargs['env'])
    env.reset(train_mode=True)

    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    kwargs['agent']['state_dim'] = state_dim
    kwargs['agent']['action_dim'] = action_dim

    agent = PPOAgent(**kwargs['agent'])
    alg = PPO(agent=agent, **kwargs['ppo'])
    alg.train(env, 400)


if __name__ == '__main__':
    with open('../config.json') as data_file:
        data = json.load(data_file)

    pprint(data)
    kwargs = data

    train(**kwargs)