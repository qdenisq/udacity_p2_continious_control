import torch
from torch.nn import Module, ModuleList, Linear, ReLU, Sigmoid
from torch.distributions import MultivariateNormal



class SimpleAgent(Module):
    def __init__(self, **kwargs):
        super(SimpleAgent, self).__init__()

        hidden_size = kwargs['hidden_size']
        self.linears = ModuleList([Linear(kwargs['state_dim'], hidden_size[0])])
        self.linears.extend([Linear(hidden_size[i-1], hidden_size[i]) for i in range(1, len(hidden_size))])
        self.mu = Linear(hidden_size[-1], kwargs['action_dim'])
        self.sigma = Linear(hidden_size[-1], kwargs['action_dim'])

        self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, x):
        for l in self.linears:
            x = l(x)
            x = self.relu(x)
        mu = self.mu(x)
        sigma = self.sigmoid(self.sigma(x))
        return mu, sigma

    def act(self, state):
        mu, sigma = self.forward(state)
        ms = [MultivariateNormal(mu[i, :], torch.eye(mu.shape[-1])*sigma[i, :]) for i in range(mu.shape[0])]
        action = torch.stack([m.sample() for m in ms], dim=0)
        probs = torch.stack([(ms[i].log_prob(action[i])).exp() for i in range(action.shape[0])], dim=0)
        return action.detach().cpu().numpy(), probs.detach().cpu().numpy()

    def get_prob(self, states, actions):
        mu, sigma = self.forward(states)
        ms = [MultivariateNormal(mu[i, :], torch.eye(mu.shape[-1]) * sigma[i, :]) for i in range(mu.shape[0])]
        probs = torch.stack([(ms[i].log_prob(actions[i])).exp() for i in range(actions.shape[0])], dim=0)
        return probs

