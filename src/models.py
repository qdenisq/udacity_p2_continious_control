import torch
from torch.nn import Module, ModuleList, Linear, ReLU, Sigmoid, Tanh
from torch.distributions import MultivariateNormal, Normal
from torch.autograd import Variable


def init_weights(m):
    if type(m) == Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=0.1)
        m.bias.data.fill_(0.01)


class SimplePPOAgent(Module):
    def __init__(self, **kwargs):
        super(SimplePPOAgent, self).__init__()

        torch.manual_seed(kwargs['seed'])
        torch.cuda.manual_seed(kwargs['seed'])

        hidden_size = kwargs['hidden_size']
        # actor
        self.linears = ModuleList([Linear(kwargs['state_dim'], hidden_size[0])])
        self.linears.extend([Linear(hidden_size[i-1], hidden_size[i]) for i in range(1, len(hidden_size))])
        self.mu = Linear(hidden_size[-1], kwargs['action_dim'])
        # self.log_var = Linear(hidden_size[-1], kwargs['action_dim'])
        self.log_var = Variable(torch.ones(kwargs['action_dim'])*-1.5, requires_grad=True)

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
            x = self.relu(x)
        mu = self.tanh(self.mu(x))
        # log_var = -2. -self.relu(self.log_var(x))
        log_var = self.log_var

        x = input
        for l in self.linears_critic:
            x = self.relu(l(x))
        v = self.v(x)
        return mu, log_var, v

    def act(self, state):
        mu, log_var, v = self.forward(state)
        sigmas = log_var.exp().sqrt() + 1e-10
        dists = Normal(mu, sigmas)
        actions = dists.sample()
        actions = torch.clamp(actions, -1., 1.)
        log_probs_0 = dists.log_prob(actions)
        log_probs = log_probs_0.sum(dim=-1)
        return actions.detach().cpu().numpy(), log_probs.detach().cpu().numpy(), v.detach().cpu().numpy()

    def get_prob_and_v(self, states, actions):
        mu, log_var, v = self.forward(states)
        sigmas = log_var.exp().sqrt() + 1e-10
        dists = Normal(mu, sigmas)
        log_probs = dists.log_prob(actions).sum(dim=-1)
        return log_probs, v, mu, sigmas

    def get_parameters(self):
        return [*self.linears.parameters(), *self.mu.parameters(), self.log_var, *self.linears_critic.parameters(), *self.v.parameters()]
        # return list(self.linears.parameters()) + list(self.mu.parameters()) + \
        #        list(self.log_var) + list(self.linears_critic.parameters()) + list(self.v.parameters())



class SimpleDDPGAgent(Module):
    def __init__(self, **kwargs):
        super(SimpleDDPGAgent, self).__init__()

        torch.manual_seed(kwargs['seed'])
        torch.cuda.manual_seed(kwargs['seed'])

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

