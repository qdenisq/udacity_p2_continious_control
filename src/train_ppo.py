import json
from pprint import pprint
import numpy as np
from src.environment import ReacherEnvironment
from src.ppo import PPO
from src.models import SimplePPOAgent
import torch
import numpy as np
import random
import datetime
import matplotlib.pyplot as plt

def train(*args, **kwargs):
    print(kwargs)

    device = 'cpu'
    kwargs['ppo']['device'] = device

    # torch.manual_seed(kwargs['agent']['seed'])
    # np.random.seed(kwargs['agent']['seed'])
    # random.seed(kwargs['agent']['seed'])

    env = ReacherEnvironment(**kwargs['env'])
    env.reset(train_mode=True)

    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    kwargs['agent']['state_dim'] = state_dim
    kwargs['agent']['action_dim'] = action_dim

    agent = SimplePPOAgent(**kwargs['agent']).to(device)
    alg = PPO(agent=agent, **kwargs['ppo'])
    scores = alg.train(env, 200)

    dt = str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p"))
    model_fname = "../models/ppo_reacher_{}.pt".format(dt)
    torch.save(agent, model_fname)

    scores_fname = "../reports/ppo_reacher_{}".format(dt)
    np.save(scores_fname, np.asarray(scores))

    plt.plot(scores)
    plt.plot(np.convolve(scores, np.ones(100)/100)[:200])
    fig_name = "../reports/ppo_reacher_{}.png".format(dt)
    plt.savefig(fig_name)


if __name__ == '__main__':
    with open('../config.json') as data_file:
        kwargs = json.load(data_file)
    pprint(kwargs)
    train(**kwargs)