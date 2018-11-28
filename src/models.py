import torch
from torch.nn import Module, ModuleList, Linear, ReLU, Sigmoid, Tanh, BatchNorm1d
from torch.distributions import MultivariateNormal, Normal
from torch.autograd import Variable


def init_weights(m):
    if type(m) == Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        # m.bias.data.fill_(0.01)


class SimplePPOAgent(Module):
    def __init__(self, **kwargs):
        super(SimplePPOAgent, self).__init__()
        hidden_size = kwargs['hidden_size']
        # actor
        self.actor_bn = BatchNorm1d(kwargs['state_dim'])
        self.actor_linears = ModuleList([Linear(kwargs['state_dim'], hidden_size[0])])
        self.actor_linears.extend([Linear(hidden_size[i - 1], hidden_size[i]) for i in range(1, len(hidden_size))])
        self.mu = Linear(hidden_size[-1], kwargs['action_dim'])
        self.log_var = Linear(hidden_size[-1], kwargs['action_dim'])
        # self.log_var = torch.nn.Parameter(torch.zeros(kwargs['action_dim']))

        # critic
        self.critic_bn = BatchNorm1d(kwargs['state_dim'])
        self.critic_linears = ModuleList([Linear(kwargs['state_dim'], hidden_size[0])])
        self.critic_linears.extend([Linear(hidden_size[i - 1], hidden_size[i]) for i in range(1, len(hidden_size))])
        self.v = Linear(hidden_size[-1], 1)

        self.relu = ReLU()
        self.tanh = Tanh()

        self.apply(init_weights) # xavier uniform init
        self.eval()

        # torch.nn.init.xavier_uniform_(self.log_var.weight, gain=0.01)

    def forward(self, state, action=None):
        x = state
        x = self.actor_bn(x)
        for l in self.actor_linears:
            x = l(x)
            x = self.relu(x)
        mu = self.tanh(self.mu(x))
        log_var = -self.relu(self.log_var(x))
        sigmas = log_var.exp().sqrt()
        dists = Normal(mu, sigmas)
        if action is None:
            action = dists.sample()
        log_prob = dists.log_prob(action).sum(dim=-1, keepdim=True)

        x = state
        x = self.critic_bn(x)
        for l in self.critic_linears:
            x = l(x)
            x = self.relu(x)
        v = self.v(x)
        return action, log_prob, dists.entropy(), v

    def get_actor_parameters(self):
        return [*self.actor_bn.parameters(), *self.actor_linears.parameters(), *self.mu.parameters(), *self.log_var.parameters()]

    def get_critic_parameters(self):
        return [*self.critic_bn.parameters(), *self.critic_linears.parameters(), *self.v.parameters()]


class SimpleDDPGAgent(Module):
    def __init__(self, **kwargs):
        super(SimpleDDPGAgent, self).__init__()

        # torch.manual_seed(kwargs['seed'])
        # torch.cuda.manual_seed(kwargs['seed'])

        hidden_size = kwargs['hidden_size']
        # actor
        self.actor_linears = ModuleList([Linear(kwargs['state_dim'], hidden_size[0])])
        self.actor_linears.extend([Linear(hidden_size[i - 1], hidden_size[i]) for i in range(1, len(hidden_size))])
        self.action = Linear(hidden_size[-1], kwargs['action_dim'])

        # critic
        self.critic_linears = ModuleList([Linear(kwargs['state_dim'] + kwargs['action_dim'], hidden_size[0])])
        self.critic_linears.extend([Linear(hidden_size[i - 1], hidden_size[i]) for i in range(1, len(hidden_size))])
        self.q = Linear(hidden_size[-1], 1)

        self.relu = ReLU()
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()

        self.apply(init_weights)  # xavier uniform init

    def act(self, state):
        x = state
        for l in self.actor_linears:
            x = l(x)
            x = self.relu(x)
        action = self.tanh(self.action(x))
        return action

    def Q(self, state, action):
        x = torch.cat([state, action], dim=1)
        for l in self.critic_linears:
            x = l(x)
            x = self.relu(x)
        q = self.q(x)
        return q

    def get_actor_parameters(self):
        return list(self.actor_linears.parameters()) + list(self.action.parameters())

    def get_critic_parameters(self):
        return list(self.critic_linears.parameters()) + list(self.q.parameters())

