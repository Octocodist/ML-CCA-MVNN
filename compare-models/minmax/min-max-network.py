import torch
import torch.nn as nn

# The classic minmax network has a fixed layer size of 4
class Group(nn.Module):
    def __init__(self, in_features, out_features):
        super(Group, self).__init__()
        self.layer = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.layer(x)
        x = torch.min(x, dim=1, keepdim=True)[0]
        return x

class MonotoneMinMax(nn.Module):
    def __init__(self, in_features, num_groups, group_size, final_output_size=5):
        super(MonotoneMinMax, self).__init__()
        self.bias = nn.Parameter(torch.ones(1))
        self.groups = nn.ModuleList([Group(in_features, group_size) for _ in range(num_groups)])
        layers = []
        #for i in range(len(hidden_sizes) - 1):
        #    layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        #    layers.append(nn.ReLU())
        #self.hidden_layers = nn.Sequential(*layers)
        #self.final_layer = nn.Linear(hidden_sizes[-1], final_output_size)
        self.final_layer(nn.Linear(num_groups, final_output_size))
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        group_outputs = [group(x) for group in self.groups]
        x = torch.cat(group_outputs, dim=1)
        x = self.hidden_layers(x)
        # max val has dim (batch_size, 1)
        max_val, _ = torch.max(self.final_activation(self.final_layer(x)), dim=1, keepdim=True)
        return max_val



