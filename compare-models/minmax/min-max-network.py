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

class MinMax(nn.Module):
    def __init__(self, in_features, num_groups, group_size, final_output_size=5):
        super(MinMax, self).__init__()
        self.bias = nn.Parameter(torch.ones(1))
        self.groups = nn.ModuleList([Group(in_features, group_size) for _ in range(num_groups)])
        layers = []
        #for i in range(len(hidden_sizes) - 1):
        #    layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        #    layers.append(nn.ReLU())
        #self.hidden_layers = nn.Sequential(*layers)
        #self.final_layer = nn.Linear(hidden_sizes[-1], final_output_size)
        self.final_layer(nn.Linear(num_groups, final_output_size))
        # this is only needed  for classification tasks
        #self.final_activation = nn.Sigmoid()

    def forward(self, x):
        x = torch.cat([x, self.bias], dim=1)
        group_outputs = [group(x) for group in self.groups]
        x = torch.cat(group_outputs, dim=1)
        x = self.hidden_layers(x)
        # max val has dim (batch_size, 1)
        #max_val, _ = torch.max(self.final_activation(self.final_layer(x)), dim=1, keepdim=True)
        max_val, _ = torch.max(self.final_layer(x), dim=1, keepdim=True)
        return max_val



class MonotoneGroup(nn.Module):
    def __init__(self, in_features, out_features, mono_mode):
        super(MonotoneGroup, self).__init__()
        if mono_mode == 'exp':
            self.trafo = torch.exp
        elif mono_mode == 'x2':
            self.trafo = torch.square
        elif mono_mode == 'weights':
            self.trafo = torch.nn.identity
        self.layer = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.trafo(x)
        x = self.layer(x)
        x = torch.min(x, dim=1, keepdim=True)[0]
        return x
    def set_weights(self):
        # set all weights of the layer to be greater of equal to zero
        weights = self.layer.weight.data
        weights[weights < 0] = 0
        self.layer.weight.data = weights



class MonotoneMinMax(nn.Module):
    def __init__(self, mono_mode, mono_in_features, mono_num_groups, mono_group_size, non_mono_in_features= 0 , non_mono_num_groups = 0, non_mono_group_size = 0 , final_output_size=5):
        super(MonotoneMinMax, self).__init__()
        self.mono_mode = mono_mode
        self.mono_bias = nn.Parameter(torch.ones(1))
        self.mono_groups = nn.ModuleList([MonotoneGroup(mono_in_features, mono_group_size, mono_mode) for _ in range(mono_num_groups)])
        self.final_input_size = mono_num_groups
        if non_mono_in_features > 0:
            self.non_mono_bias = nn.Parameter(torch.ones(1))
            self.non_mono_groups = nn.ModuleList([Group(non_mono_in_features, non_mono_group_size) for _ in range(non_mono_num_groups)])
            self.final_input_size += non_mono_num_groups

        self.final_layer(nn.Linear(self.final_input_size, final_output_size))

    def forward(self, x_mono, x_non_mono=None):

        x_mono = torch.cat([x_mono, self.bias], dim=1)
        mono_group_outputs = [group(x_mono) for group in self.mono_groups]
        x_mono = torch.cat(mono_group_outputs, dim=1)
        if x_non_mono is not None:
            x_non_mono = torch.cat([ x_non_mono, self.non_mono_bias], dim=1)
            non_mono_group_outputs = [group(x_non_mono) for group in self.groups]
            x_non_mono = torch.cat(non_mono_group_outputs, dim=1)
            x_mono = torch.cat([x_mono, x_non_mono], dim=1)
        x = self.hidden_layers(x_mono)
        max_val, _ = torch.max(self.final_layer(x), dim=1, keepdim=True)
        return max_val
