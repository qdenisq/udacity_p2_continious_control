import torch
from torch.nn import Module, ModuleList, Linear, ReLU, Sigmoid, Tanh
from torch.distributions import MultivariateNormal, Normal


def init_weights(m):
    if type(m) == Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class SimpleAgent(Module):
    def __init__(self, **kwargs):
        super(SimpleAgent, self).__init__()

        torch.manual_seed(kwargs['seed'])
        torch.cuda.manual_seed(kwargs['seed'])

        hidden_size = kwargs['hidden_size']
        # actor
        self.linears = ModuleList([Linear(kwargs['state_dim'], hidden_size[0])])
        self.linears.extend([Linear(hidden_size[i-1], hidden_size[i]) for i in range(1, len(hidden_size))])
        self.mu = Linear(hidden_size[-1], kwargs['action_dim'])
        self.log_var = Linear(hidden_size[-1], kwargs['action_dim'])

        # critic
        self.linears_critic = ModuleList([Linear(kwargs['state_dim'], hidden_size[0])])
        self.linears_critic.extend([Linear(hidden_size[i - 1], hidden_size[i]) for i in range(1, len(hidden_size))])
        self.v = Linear(hidden_size[-1], 1)

        self.relu = ReLU()
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()

        self.apply(init_weights)  # xavier uniform init
        # torch.nn.init.xavier_uniform_(self.log_var.weight, gain=0.01)

    def forward(self, input):
        x = input
        for l in self.linears:
            x = l(x)
            x = self.tanh(x)
        mu = self.tanh(self.mu(x))
        log_var = -2. -self.relu(self.log_var(x))

        x = input
        for l in self.linears_critic:
            x = self.tanh(l(x))
        v = self.v(x)
        return mu, log_var, v

    def act(self, state):
        mu, log_var, v = self.forward(state)
        sigmas = log_var.exp().sqrt() + 1e-5
        dists = Normal(mu, sigmas)
        actions = dists.sample()
        actions = torch.clamp(actions, -1., 1.)
        log_probs = dists.log_prob(actions).sum(dim=-1)

        return actions.detach().cpu().numpy(), log_probs.detach().cpu().numpy(), v.detach().cpu().numpy()

    def get_prob_and_v(self, states, actions):
        mu, log_var, v = self.forward(states)
        sigmas = log_var.exp().sqrt() + 1e-5
        dists = Normal(mu, sigmas)
        log_probs = dists.log_prob(actions).sum(dim=-1)
        return log_probs, v, mu, sigmas



