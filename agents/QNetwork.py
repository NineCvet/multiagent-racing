import torch.nn as nn
import torch.nn.init as init
import math


class QNetwork(nn.Module):
    def __init__(self, input_size, num_actions, hidden_sizes=(64, 64), device='cuda'):
        super(QNetwork, self).__init__()

        # Defining layers
        layers = [nn.Linear(input_size, hidden_sizes[0]), nn.ReLU()]
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], num_actions))

        # Creating a Sequential model
        self.model = nn.Sequential(*layers)

        # Moving model to the specified device
        self.device = device
        self.to(device)
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights using He initialization."""
        for m in self.model:
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, a=math.sqrt(5))  # He initialization
                if m.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        x = x.to(self.device)  # Ensuring input is on the right device
        return self.model(x)
