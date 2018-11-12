import json
from pprint import pprint
import numpy as np
from src.environment import ReacherEnvironment
from src.ppo import PPO
from src.models import SimpleAgent

def train(*args, **kwargs):
    print(kwargs)

    env = ReacherEnvironment(**kwargs['env'])
    env.reset(train_mode=True)

    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    kwargs['agent']['state_dim'] = state_dim
    kwargs['agent']['action_dim'] = action_dim



    agent = SimpleAgent(**kwargs['agent'])
    alg = PPO(agent=agent, **kwargs['ppo'])
    alg.train(env, 400)


    num_agents = env.get_num_agents()
    action_size = env.get_action_dim()
    for i in range(2):
        done = False
        score = 0
        state = env.reset(train_mode=True)
        while not np.any(done):
            action = np.random.randn(num_agents, action_size)  # select an action (for each agent)
            action = np.clip(action, -1, 1)
            env.step(action)
            state, reward, done = env.step(action)  # roll out transition
            score += np.mean(reward)
            print("\r play #{}, reward: {} | score: {}".format(i+1, reward, score), end='')
        print()


if __name__ == '__main__':
    with open('../config.json') as data_file:
        data = json.load(data_file)

    pprint(data)
    kwargs = data

    train(**kwargs)