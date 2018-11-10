import torch
from torch.nn import Module, ModuleList, Linear, ReLU, Sigmoid
from torch.distributions import MultivariateNormal, Normal



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
        log_var = self.sigma(x)
        return mu, log_var

    def act(self, state):
        mu, log_var = self.forward(state)
        sigmas = log_var.exp().sqrt() + 1e-5
        dists = Normal(mu, sigmas)
        actions = dists.sample()
        actions = torch.clamp(actions, -1, 1)
        log_probs = dists.log_prob(actions).sum(dim=-1)
        return actions.detach().cpu().numpy(), log_probs.detach().cpu().numpy()

    def get_prob(self, states, actions):
        mu, log_var = self.forward(states)
        sigmas = log_var.exp().sqrt()+ 1e-5
        dists = Normal(mu, sigmas)
        log_probs = dists.log_prob(actions).sum(dim=-1)
        return log_probs

