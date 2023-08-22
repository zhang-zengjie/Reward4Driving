import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class QNetworkL(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed=32, hidden_units=[512, 256, 128, 64]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetworkL, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.fc1 = nn.Linear(96*96, hidden_units[0]) # (96,96), 144
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.fc4 = nn.Linear(hidden_units[2], hidden_units[3])
        # self.fc5 = nn.Linear(hidden_units[3], action_size)

        # self.seq = nn.Sequential(

        #     nn.Linear(96, 96),
        #     nn.ReLU(inplace=True),

        #     nn.Linear(96, hidden_units[0]),
        #     nn.ReLU(inplace=True),

        #     nn.Linear(hidden_units[0], hidden_units[0]),
        #     nn.ReLU(inplace=True),

        #     nn.Linear(hidden_units[0], hidden_units[1]),
        #     nn.ReLU(inplace=True),

        #     nn.Linear(hidden_units[1], hidden_units[2]),
        #     nn.ReLU(inplace=True),
        # )
        self.fc5 = nn.Linear(hidden_units[3], action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        state = state.reshape(-1,96*96)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))        
        out = self.fc5(x)

        return out


class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise
    N.B. nn.Linear already initializes weight and bias to
    """
    def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
        super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
        if bias:
            self.sigma_bias = nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))

    def forward(self, input):
        bias = self.bias
        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))

        with torch.no_grad():
            torch.randn(self.epsilon_input.size(), out=self.epsilon_input)
            torch.randn(self.epsilon_output.size(), out=self.epsilon_output)
            eps_in = func(self.epsilon_input)
            eps_out = func(self.epsilon_output)
            noise_v = torch.mul(eps_in, eps_out).detach()
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)
