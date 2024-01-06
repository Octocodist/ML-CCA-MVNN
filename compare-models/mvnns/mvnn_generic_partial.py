import numpy as np
import torch
from torchinfo import summary
import torch.nn as nn
from mvnns.layers import *
from mvnns.mvnn_generic import MVNN_GENERIC

class MVNN_GENERIC_PARTIAL(nn.Module):

    def __init__(self,
                 input_dim: int,
                 num_hidden_layers: int,
                 num_hidden_units: int,
                 dropout_prob: float,
                 layer_type: str,
                 target_max: float,
                 init_method: str,
                 random_ts: tuple,
                 trainable_ts: bool,
                 init_E: float,
                 init_Var: float,
                 init_b: float,
                 init_bias: float,
                 init_little_const: float,
                 lin_skip_connection: bool,
                 capacity_generic_goods: np.array,
                 output_inner_mvnn: int,

                 non_mono_input_dim : int,
                 non_mono_num_hidden_layers : int,
                 non_mono_num_hidden_units : int,
                 non_mono_output_dim : float,
                 non_mono_lin_skip_connection: bool,
                 non_mono_dropout_prob: float,

                 #final_input_dim: int,
                 final_num_hidden_layers: int,
                 final_num_hidden_units: int,
                 final_dropout_prob: float,
                 final_layer_type: str,
                 final_target_max: float,
                 final_init_method: str,
                 final_random_ts: tuple,
                 final_trainable_ts: bool,
                 final_init_E: float,
                 final_init_Var: float,
                 final_init_b: float,
                 final_init_bias: float,
                 final_init_little_const: float,
                 final_lin_skip_connection: bool,
                 final_capacity_generic_goods: np.array,
                 final_output_inner_mvnn: int, # this is the final output

                 *args, **kwargs):

        super(MVNN_GENERIC_PARTIAL, self).__init__()

        self.mvnn_input = MVNN_GENERIC(input_dim,num_hidden_layers, num_hidden_units,dropout_prob,layer_type,target_max,init_method,random_ts,trainable_ts,init_E,init_Var,init_b,init_bias,init_little_const,lin_skip_connection,capacity_generic_goods, output_inner_mvnn)
        #self.intermediate = nn.Linear(output_inner_mvnn + non_mono_output_dim*2 ,final_input_dim,bias=True)
        # this mvnn must have at least 3 hidden layers
        print( "dimensions are:  ", input_dim, non_mono_input_dim, output_inner_mvnn, non_mono_output_dim, final_output_inner_mvnn)
        final_input_dims = output_inner_mvnn + non_mono_output_dim
        print("final output is : ", final_input_dims) 
        self.mvnn_final_1 = MVNN_GENERIC(final_input_dims, final_num_hidden_layers,final_num_hidden_units,final_dropout_prob, final_layer_type,final_target_max,final_init_method,final_random_ts,final_trainable_ts,final_init_E,final_init_Var,final_init_b,final_init_bias,final_init_little_const,final_lin_skip_connection,final_capacity_generic_goods,final_output_inner_mvnn)
        self.non_mono_layers = []

        #initial layer
        self.non_mono_layers.append(
            nn.Linear(non_mono_input_dim,
                      non_mono_num_hidden_units,
                      bias=True
                      )
        )
        self.non_mono_layers.append(nn.ReLU())
        # inner layers
        for _ in range(non_mono_num_hidden_layers):
            self.non_mono_layers.append(
                nn.Linear(non_mono_num_hidden_units,
                         non_mono_num_hidden_units,
                         bias=True
                         ))
            self.non_mono_layers.append(nn.ReLU())
            
        self.non_mono_list = torch.nn.ModuleList(self.non_mono_layers)
        self.non_mono_dropouts = torch.nn.ModuleList([nn.Dropout(p=non_mono_dropout_prob) for _ in range(len(self.non_mono_list))])

        self.non_mono_net = torch.nn.Sequential(*self.non_mono_list)

        # final layer # keep this separate for skip connection
        self.non_mono_output_layer = (
            nn.Linear(non_mono_num_hidden_units,
                      non_mono_output_dim,
                      bias=True
                      )
        )
        self.non_mono_output_activation= nn.ReLU()

        if non_mono_lin_skip_connection:
            self.lin_skip_layer = nn.Linear(non_mono_input_dim,non_mono_output_dim,bias=False)

    def forward(self, x_mono, x_non_mono):

        print("shape x_mono: ", x_mono.shape) 
        x_mono = self.mvnn_input(x_mono)

        if hasattr(self, 'non_mono_lin_skip_layer'):
            x_in_non_mono = x_non_mono

        # Non-Monotonic part
        x_non_mono = self.non_mono_net(x_non_mono)
        #for layer, dropout in zip(self.non_mono_layers, self.non_mono_dropouts):
        #    x_non_mono = layer(x_non_mono)
        #    x_non_mono = dropout(x_non_mono)

        # Output layer
        if hasattr(self, 'non_mono_lin_skip_layer'):
            x_non_mono = self.non_mono_output_activation(self.non_mono_output_layer(x_non_mono)) + self.lin_skip_layer(x_in_non_mono)
        else:
            x_non_mono = self.non_mono_output_activation(self.non_mono_output_layer(x_non_mono))
        x_middle = torch.cat((x_mono,x_non_mono),dim=1)

        print(" middle shape: ", x_middle.shape)
        x_final = self.mvnn_final_1(x_middle)

        return x_final
    def transform_weights(self):
        self.mvnn_input.transform_weights()
        self.mvnn_final.transform_weights()

        """
        fc_layer = eval(layer_type)
        

        self.output_activation_function = torch.nn.Identity()
        self._layer_type = layer_type
        self._num_hidden_layers = num_hidden_layers
        self._target_max = target_max
        self.capacity_generic_goods = capacity_generic_goods

        self.layers = []

        # NEW: Generic Transformation with requires_grad=False
        #------------------------
        generic_trafo_layer = nn.Linear(in_features = input_dim,
                                        out_features = input_dim,
                                        bias = False
                                        )
        
        generic_trafo_layer_weight = np.diag(1/self.capacity_generic_goods)
        generic_trafo_layer_weight = generic_trafo_layer_weight.astype(np.float32)
        generic_trafo_layer.weight.data = torch.from_numpy(generic_trafo_layer_weight)

        for param in generic_trafo_layer.parameters():
            param.requires_grad = False


        self.layers.append(generic_trafo_layer)
        #------------------------


        fc1 = fc_layer(input_dim,
                       num_hidden_units,
                       init_method=init_method,
                       random_ts=random_ts,
                       trainable_ts=trainable_ts,
                       use_brelu=True,
                       bias=True,
                       init_E=init_E,
                       init_Var=init_Var,
                       init_b=init_b,
                       init_bias=init_bias,
                       init_little_const=init_little_const
                       )

        self.layers.append(fc1)
        for _ in range(num_hidden_layers - 1):
            self.layers.append(
                fc_layer(num_hidden_units,
                         num_hidden_units,
                         init_method=init_method,
                         random_ts=random_ts,
                         trainable_ts=trainable_ts,
                         use_brelu=True,
                         bias=True,
                         init_E=init_E,
                         init_Var=init_Var,
                         init_b=init_b,
                         init_bias=init_bias,
                         init_little_const=init_little_const
                         )
            )

        self.layers = torch.nn.ModuleList(self.layers)
        self.dropouts = torch.nn.ModuleList([nn.Dropout(p=dropout_prob) for _ in range(len(self.layers))])

        self.output_layer = fc_layer(num_hidden_units,
                                     1,
                                     init_method=init_method,
                                     random_ts=random_ts,
                                     trainable_ts=trainable_ts,
                                     bias=False,
                                     use_brelu=False,
                                     init_E=init_E,
                                     init_Var=init_Var,
                                     init_b=init_b,
                                     init_bias=init_bias,
                                     init_little_const=init_little_const
                                     )
        if lin_skip_connection:
            self.lin_skip_layer = fc_layer(input_dim,
                                           1,
                                           init_method='zero',
                                           random_ts=None,
                                           trainable_ts=False,
                                           bias=False,
                                           use_brelu=False,
                                           init_E=None,
                                           init_Var=None,
                                           init_b=None,
                                           init_bias=None,
                                           init_little_const=None
                                           )
        self.dataset_info = None

    def forward(self, x):
        if hasattr(self, 'lin_skip_layer'):
            x_in = x
        for layer, dropout in zip(self.layers, self.dropouts):
            x = layer(x)
            x = dropout(x)

        # Output layer
        if hasattr(self, 'lin_skip_layer'):
            x = self.output_activation_function(self.output_layer(x)) + self.lin_skip_layer(x_in)
        else:
            x = self.output_activation_function(self.output_layer(x))
        return x

    def set_dropout_prob(self, dropout_prob):
        for dropout in self.dropouts:
            dropout.p = dropout_prob

    def transform_weights(self):
        for layer in self.layers:
            if hasattr(layer, 'transform_weights'):
                layer.transform_weights()
        if hasattr(self.output_layer, 'transform_weights'):
            self.output_layer.transform_weights()

    
    def print_parameters(self):
        i = 0
        for layer in self.layers:
            print(f'Layer {i}: {layer}')
            print('layer.weight')
            print(f'Shape: {layer.weight.data.shape}')
            print(f'Values: {layer.weight.data}')
            if layer.bias is not None:
                print('layer.bias')
                print(f'Shape: {layer.bias.data.shape}')
                print(f'Values: {layer.bias.data}')
            for name, param in layer.named_parameters():
                print(f'{name} requires_grad={param.requires_grad}')
            i += 1
            print()
        print(f'Output Layer')
        print('output_layer.weight')
        print(f'Shape: {self.output_layer.weight.data.shape}')
        print(f'Values: {self.output_layer.weight.data}')
        if self.output_layer.bias is not None:
                print('output_layer.bias.bias')
                print(f'Shape: {self.output_layer.bias.data.shape}')
                print(f'Values: {self.output_layer.bias.data}')
        for name, param in self.output_layer.named_parameters():
                print(f'{name} requires_grad={param.requires_grad}')
        """
