import os
import multiprocessing as mp
import numpy
import pickle
import torch.nn as nn
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics as skm
from torch.optim import Adam
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset
import torch.nn.functional as F

from scipy.stats import kendalltau
import wandb

### custom imports ###
from mvnns.mvnn import MVNN
from mvnns.mvnn_generic import MVNN_GENERIC
from generalized_UMNN.models.MultidimensionnalMonotonicNN import SlowDMonotonicNN

from minmax.min_max_network import MonotoneMinMax
from minmax.min_max_network import MinMax
import CertifiedMonotonicNetwork

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import pandas as pd
import torch.utils.data as Data
import argparse
from torch.utils.data import DataLoader
from certify import *
#from config import cert_parameters, mvnn_parameters, umnn_parameters
from mvnns.mvnn_generic_partial import MVNN_GENERIC_PARTIAL



def init_parser():
    parser = argparse.ArgumentParser()

    ### experiment parameters ###
    #parser.add_argument("--experiment", help="experiment to run", default="0", choices=['0','1'])
    parser.add_argument("--dataset", help="dataset to use", default="blog", choices=['gsvm' , 'lsvm', 'srvm', 'mrvm', 'blog', 'compas'] )
    parser.add_argument("--nbids", help="number of bids to use", default=25000)
    parser.add_argument("--bidder_id", help="bidder id to use", default=0)
    parser.add_argument('-m','--model',  type=str, help='Choose model to train: UMNN, MVNN', choices=['UMNN','MVNN','CERT', "PMVNN", "PCERT", "PUMNN", "MINMAX", "MONOMINMAX"],
                        default='PCERT' )
    parser.add_argument('--sample',  type= bool , help='Set mode to testing ', default="False")
    parser.add_argument("-ns","--num_seeds", type=int, default=1, help="number of seeds to use for hpo")
    parser.add_argument("-is","--initial_seed", type=int, default=100, help="initial seed to use for hpo")

    parser.add_argument("-tp","--train_percent", type=float, default=0.2, help="percentage of data to use for training")
    parser.add_argument("-ud","--use_dummy", type=bool, default=True, help="use dummy dataset")
    parser.add_argument("--dropout_decay", help="dropout decay per epoch ", default=0.97)

    ### training parameters ###
    parser.add_argument("-e","--epochs", help="number of epochs to train", default=3)
    parser.add_argument("--batch_size", help="batch size to use", default=18)
    parser.add_argument("--learning_rate", help="learning rate", default=0.001)
    parser.add_argument("--l2_rate", help="l2 norm", default=0.)
    parser.add_argument("--num_train_points", help="num_training data ", default=50)
    parser.add_argument("--scale", help="scale to 0-1", type= bool, default=True)

    ### model parameters ###
    ### PMVNN and CERT
    parser.add_argument("--num_hidden_layers", help="number of hidden layers", default=3)
    parser.add_argument("--num_hidden_units", help="number of hidden units", default=30)
    parser.add_argument("--non_mono_num_hidden_layers", help="number of non mono hidden layers", default=5)
    parser.add_argument("--non_mono_num_hidden_units", help="number of non mono hidden units", default=28)

    ### for MVNN 
    parser.add_argument("--layer_type", help="layer type", default="MVNNLayerReLUProjected")
    parser.add_argument("--target_max", help="target max", default=1)
    parser.add_argument("--lin_skip_connection", type=bool,  help="linear skip connection", default=False)
    parser.add_argument("--dropout_prob", help="dropout probability", default=0.)
    parser.add_argument("--trainable_ts", help="trainable ts", default=True)
   
    ### for PMVNN
    parser.add_argument("--output_inner_mvnn", default=5)
    parser.add_argument("--non_mono_output_dim", help="number ofnon mono output units", default=12)
    parser.add_argument("--non_mono_lin_skip_connection", type=bool,  help="linear skip connection", default=False)
    parser.add_argument("--non_mono_dropout_prob", help="dropout probability", default=0.)

    parser.add_argument("--final_lin_skip_connection", type=bool,  help="linear skip connection", default=False)
    parser.add_argument("--final_num_hidden_layers", help="number of hidden layers", default=1)
    parser.add_argument("--final_num_hidden_units", help="number of hidden units", default=20)
    parser.add_argument("--final_dropout_prob", help="final dropout probability", default=0.)
    parser.add_argument("--final_trainable_ts", help="final trainable ts", default=True)
    parser.add_argument("--final_output_inner_mvnn", help="final output inner mvnn", default=10)

    # for CERT
    parser.add_argument("--compress_non_mono", help="compressing mono inputs", type= bool, default=True)
    parser.add_argument("--normalize_regression", help="normalizing regression", type= bool, default=True)
    parser.add_argument("--bottleneck", help="number of bottleneck  units", default=10)



    # for MINMAX
    parser.add_argument("--mono_mode", choices=["weights","x2","exp"], help="mono mode for minmax", type= str, default="exp")
    parser.add_argument("--num_groups",  help="num groups for minmax", type= int, default=5)
    parser.add_argument("--group_size",  help="group size for minmax", type= int, default=10)
    parser.add_argument("--final_output_size",  help="final output size before last layer", type= int, default=5)
    parser.add_argument("--non_mono_num_groups",  help="num non mono groups for minmax", type= int, default=5)
    parser.add_argument("--non_mono_group_size",  help="non_mono group size for minmax", type= int, default=10)

    return parser

#TODO: take this from a file
### default parameters ###
minmax_parameters = {
    'num_hidden_units': 20,
    'num_groups': 4,
    'group_size': 20,
    'final_output_size': 10,
}
monominmax_parameters = {
    'mono_num_hidden_units': 20,
    'mono_num_groups': 4,
    'mono_group_size': 20,
    #'final_output_size': 10,
}
mvnn_parameters = {'num_hidden_layers': 1,
                    'num_hidden_units': 20,
                    'layer_type': 'MVNNLayerReLUProjected',
                    'target_max': 1,
                    'lin_skip_connection': 1,
                    'dropout_prob': 0,
                    'init_method':'custom',
                    'random_ts': [0,1],
                    'trainable_ts': True,
                    'init_E': 1,
                    'init_Var': 0.09,
                   'init_b': 0.05,
                   'init_bias': 0.05,
                   'init_little_const': 0.1
                   }
pmvnn_parameters = {'num_hidden_layers': 2,
                           'num_hidden_units': 20,
                           'layer_type': 'MVNNLayerReLUProjected',
                           'target_max': 1,
                           'lin_skip_connection': 1,
                           'dropout_prob': 0,
                           'init_method': 'custom',
                           'random_ts': [0, 1],
                           'trainable_ts': True,
                           'init_E': 1,
                           'init_Var': 0.09,
                           'init_b': 0.05,
                           'init_bias': 0.05,
                           'init_little_const': 0.1,
                           'output_inner_mvnn': 15, 

                           'non_mono_num_hidden_layers': 2,
                           'non_mono_num_hidden_units': 20,
                           'non_mono_output_dim': 10,
                           'non_mono_lin_skip_connection': 0,
                           'non_mono_dropout_prob': 0,

                           'final_num_hidden_layers': 3,
                           'final_num_hidden_units': 20,
                           'final_dropout_prob': 0,
                           'final_layer_type': 'MVNNLayerReLUProjected',
                           'final_target_max': 1,
                           'final_init_method': 'custom',
                           'final_random_ts': [0, 1],
                           'final_trainable_ts': True,
                           'final_init_E': 1,
                           'final_init_Var': 0.09,
                           'final_init_b': 0.05,
                           'final_init_bias': 0.05,
                           'final_init_little_const': 0.1,
                           'final_lin_skip_connection': 0,
                           'final_output_inner_mvnn': 1,
                           }

umnn_parameters = {"num_embedding_layers": 1, "num_embedding_hiddens": 10, "num_main_hidden_layers" : 1, "num_main_hidden_nodes": 20, "n_out": 1,"nb_steps": 10 }
#umnn_parameters = {"mon_in": 1, "cond_in": 10, "hiddens": [20,20], "n_out": 1, "nb_steps": 50, "device": "cpu"}
cert_parameters = {"output_parameters": 1, "num_hidden_layers": 4, "hidden_nodes": 20}



def load_dataset(args, num_train_data=50, train_percent=0.2, seed=100):
    # load dataset using pickle
    # parse filepath

    if args.dataset == "lsvm" or args.dataset == "gsvm" or args.dataset == "mrvm" or args.dataset == "gsvm":
        filepath = "./dataset_generation/datasets/"+ str(args.dataset)+"/"+str(args.dataset)+"_"+str(seed)+"_"+str(args.nbids)+".pkl"
        
        with open(filepath, "rb") as file:
            dataset = pickle.load(file)
        X = dataset[0]
        y = dataset[1]
        if args.use_dummy:
            X = [bundle+(0,) for bundle in X]

        #in case train percent is not set, use num_train_data
        if train_percent == 0:
            train_percent = num_train_data/len(X)

        #do train val test split
        X_train, test_and_val_X, y_train, test_and_val_y = train_test_split(X, y, test_size=train_percent, random_state=1)
        X_val, X_test, y_val, y_test = train_test_split(test_and_val_X,test_and_val_y, test_size=0.5, random_state=1)

        # transform to tensors
        X_train_tensor = torch.FloatTensor(X_train).float()
        y_train_tensor = torch.FloatTensor(y_train).float()
        X_val_tensor = torch.FloatTensor(X_val).float()
        y_val_tensor = torch.FloatTensor(y_val).float()
        X_test_tensor = torch.FloatTensor(X_test).float()
        y_test_tensor = torch.FloatTensor(y_test).float()
        
        if args.scale: 
            max_val = max(torch.max(y_train_tensor).item(), torch.max(y_val_tensor).item(),torch.max( y_test_tensor).item())
            print(max_val, " is max_val " ) 
            y_train_tensor = torch.div(y_train_tensor, max_val)
            y_val_tensor = torch.div(y_val_tensor, max_val)
            y_test_tensor = torch.div(y_test_tensor, max_val)


        #create datasets for dataloader
        return TensorDataset(X_train_tensor, y_train_tensor), TensorDataset(X_val_tensor, y_val_tensor),TensorDataset(X_test_tensor, y_test_tensor)



    # Load Mixed Datasets
    else:     
        filepath = "./dataset_generation/experiment_2/"
        with open(filepath+ str(args.dataset)+"_train.pkl", "rb") as file: 
            dataset = pickle.load(file)
        X_non_mono_tv = dataset[0]
        X_mono_tv = dataset[1]
        y_tv = dataset[2]

        with open(filepath+str(args.dataset)+"_test.pkl","rb") as file: 
            dataset_test = pickle.load(file)
        X_non_mono_test = dataset_test[0]
        X_mono_test = dataset_test[1]
        y_test = dataset_test[2]

        if train_percent == 0:
            train_percent = num_train_data/len(X)
        # mono train val test split

        X_non_mono_train, X_non_mono_val, y_non_mono_train, y_non_mono_val = train_test_split(X_non_mono_tv, y_tv, train_size=train_percent, random_state=666)
        X_mono_train, X_mono_val, y_mono_train, y_mono_val = train_test_split(X_mono_tv, y_tv, train_size=train_percent, random_state=666)

        # these should be the same
        assert(y_non_mono_train.iloc[0,0]== y_mono_train.iloc[0,0])
        #assert(y_non_mono_val == y_mono_val)

        #print(X_non_mono_train.type) 

        # transform to tensors
        X_non_mono_train_tensor = torch.Tensor(X_non_mono_train.values)
        X_non_mono_val_tensor = torch.FloatTensor(X_non_mono_val.values)
        X_non_mono_test_tensor = torch.FloatTensor(X_non_mono_test.values)

        X_mono_train_tensor = torch.FloatTensor(X_mono_train.values)
        X_mono_val_tensor = torch.FloatTensor(X_mono_val.values)
        X_mono_test_tensor = torch.FloatTensor(X_mono_test.values)

        y_train_tensor = torch.FloatTensor(y_non_mono_train.values)
        y_val_tensor = torch.FloatTensor(y_non_mono_val.values)
        y_test_tensor = torch.FloatTensor(y_test.values)


        #y_mono_train_tensor = torch.FloatTensor(y_mono_train).float()
        #y_mono_val_tensor = torch.FloatTensor(y_mono_val).float()
        #y_test_tensor = torch.FloatTensor(y_test).float()


        if args.scale: 
            max_val =torch.max(y_train_tensor).item()
            print(max_val, " is max_val " ) 
            y_train_tensor = torch.div(y_train_tensor, max_val)
            y_val_tensor = torch.div(y_val_tensor, max_val)
            y_test_tensor = torch.div(y_test_tensor, max_val)


        #create datasets for dataloader
        return TensorDataset(X_mono_train_tensor,X_non_mono_train_tensor, y_train_tensor), TensorDataset(X_mono_val_tensor,X_non_mono_val_tensor, y_val_tensor),TensorDataset(X_mono_test_tensor,X_non_mono_test_tensor, y_test_tensor)


pumnn_parameters = {"out_embedding": 18, "num_embedding_layers": 1, "num_embedding_hiddens": 10, "num_main_hidden_layers" : 1, "num_main_hidden_nodes": 20, "n_out": 1,"nb_steps": 10 , "out_embedding_same": True}

### UMNN Section ###
#This network needs an embedding Network and a umnn network
#TODO change this from hardcoded
class PartialEmbeddingNet(nn.Module):
    def __init__(self, in_embedding, in_main, out_embedding, device='cpu',num_embedding_layers=3, num_embedding_hiddens=200, num_main_hidden_layers=3, num_main_hidden_nodes=100, nb_steps=10, dropout_prob=0.):
        super(PartialEmbeddingNet, self).__init__()
        ## Attention this dynamic setting of embedding was modified from the original code
        self.layers = []
        self.layers = [nn.Linear(in_embedding, num_embedding_hiddens), nn.ReLU()]
        for i in range (num_embedding_layers):
            self.layers.append(nn.Linear(num_embedding_hiddens, num_embedding_hiddens))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(num_embedding_hiddens, out_embedding))
        self.layers.append(nn.ReLU())
        self.embedding_net = nn.Sequential(*self.layers).to(device)

        self.dropouts = nn.Dropout(p=dropout_prob)

        self.umnn_hidden = []
        for i in range(num_main_hidden_layers):
            self.umnn_hidden.append(num_main_hidden_nodes)
        self.umnn = SlowDMonotonicNN(in_main, cond_in=out_embedding, hiddens=self.umnn_hidden, n_out=1, nb_steps= nb_steps, device= device)
        #self.embedding_net = nn.Sequential(nn.Linear(in_embedding, 200), nn.ReLU(),
        #                                   nn.Linear(200, 200), nn.ReLU(),
        #                                   nn.Linear(200, out_embedding), nn.ReLU()).to(device)
        #self.umnn = SlowDMonotonicNN(in_main, cond_in=out_embedding, hiddens=[100, 100, 100], n_out=1, nb_steps= 300, device= device)

    def set_steps(self, nb_steps):
        self.umnn.set_steps(nb_steps)

    def forward(self, x_mono, x_non_mono):

        h = self.embedding_net(x_non_mono)
        
        x = torch.nan_to_num(torch.sigmoid(self.umnn(x_mono, h)))
        x = self.dropouts(x)
        # CARE THIS IS A MESSY QUICK FIX 

        return x
    def decay_dropout(self, rate):
        self.dropouts.p = self.dropouts.p * rate

class EmbeddingNet(nn.Module):
    def __init__(self, in_embedding, in_main, out_embedding, device='cpu',num_embedding_layers=3, num_embedding_hiddens=200, num_main_hidden_layers=3, num_main_hidden_nodes=100, nb_steps=10):
        super(EmbeddingNet, self).__init__()
        ## Attention this dynamic setting of embedding was modified from the original code
        self.layers = []
        self.layers = [nn.Linear(in_embedding, num_embedding_hiddens), nn.ReLU()]
        for i in range (num_embedding_layers):
            self.layers.append(nn.Linear(num_embedding_hiddens, num_embedding_hiddens))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(num_embedding_hiddens, out_embedding))
        self.layers.append(nn.ReLU())
        self.embedding_net = nn.Sequential(*self.layers).to(device)
        self.umnn_hidden = []
        for i in range(num_main_hidden_layers):
            self.umnn_hidden.append(num_main_hidden_nodes)
        self.umnn = SlowDMonotonicNN(in_main, cond_in=out_embedding, hiddens=self.umnn_hidden, n_out=1, nb_steps= nb_steps, device= device)
        #self.embedding_net = nn.Sequential(nn.Linear(in_embedding, 200), nn.ReLU(),
        #                                   nn.Linear(200, 200), nn.ReLU(),
        #                                   nn.Linear(200, out_embedding), nn.ReLU()).to(device)
        #self.umnn = SlowDMonotonicNN(in_main, cond_in=out_embedding, hiddens=[100, 100, 100], n_out=1, nb_steps= 300, device= device)

    def set_steps(self, nb_steps):
        self.umnn.set_steps(nb_steps)

    def forward(self, x):
        h = self.embedding_net(x[:,:])
        #h = self.embedding_net(x)
        return torch.sigmoid(self.umnn(x, h))


### CERT Section ###
def generate_regularizer(in_list, out_list):
    length = len(in_list)
    reg_loss = 0.
    min_derivative = 0.0
    wandb.config.update({'min_derivative': min_derivative})

    for i in range(length):
        xx = in_list[i]
        yy = out_list[i]
        for j in range(yy.shape[1]):
            grad_input = torch.autograd.grad(torch.sum(yy[:, j]), xx, create_graph=True, allow_unused=True)[0]
            grad_input_neg = -grad_input
            grad_input_neg += .2
            grad_input_neg[grad_input_neg < 0.] = 0.
            if min_derivative < torch.max(grad_input_neg ** 2):
                min_derivative = torch.max(grad_input_neg ** 2)
    reg_loss = min_derivative
    return reg_loss

class MLP_relu(nn.Module):
    def __init__(self, mono_feature, non_mono_feature, mono_sub_num=1, non_mono_sub_num=1, mono_hidden_num=5,
                 non_mono_hidden_num=5, compress_non_mono=False, normalize_regression=False, dropout_prob=0., bottleneck=10.):
        super(MLP_relu, self).__init__()
        self.lam = 10
        self.normalize_regression = normalize_regression
        self.compress_non_mono = compress_non_mono

        self.mono_dropouts = torch.nn.ModuleList([nn.Dropout(p=dropout_prob) for _ in range(mono_sub_num)])
        self.non_mono_dropouts = torch.nn.ModuleList([nn.Dropout(p=dropout_prob) for _ in range(non_mono_sub_num)])

        if compress_non_mono  :
            self.non_mono_feature_extractor = nn.Linear(non_mono_feature, bottleneck, bias=True)
            self.mono_fc_in = nn.Linear(mono_feature + bottleneck, mono_hidden_num, bias=True)
        else:
            self.mono_fc_in = nn.Linear(mono_feature + non_mono_feature, mono_hidden_num, bias=True)

        self.non_mono_fc_in = nn.Linear(non_mono_feature, non_mono_hidden_num, bias=True)
        self.mono_submods_out = nn.ModuleList(
            [nn.Linear(mono_hidden_num, bottleneck, bias=True) for i in range(mono_sub_num)])
        self.mono_submods_in = nn.ModuleList(
            [nn.Linear(2 * bottleneck, mono_hidden_num, bias=True) for i in range(mono_sub_num)])
        self.non_mono_submods_out = nn.ModuleList(
            [nn.Linear(non_mono_hidden_num, bottleneck, bias=True) for i in range(mono_sub_num)])
        self.non_mono_submods_in = nn.ModuleList(
            [nn.Linear(bottleneck, non_mono_hidden_num, bias=True) for i in range(mono_sub_num)])

        self.mono_fc_last = nn.Linear(mono_hidden_num, 1, bias=True)
        self.non_mono_fc_last = nn.Linear(non_mono_hidden_num, 1, bias=True)

    def forward(self, mono_feature, non_mono_feature):
        y = self.non_mono_fc_in(non_mono_feature)
        y = F.relu(y)

        if self.compress_non_mono:
            non_mono_feature = self.non_mono_feature_extractor(non_mono_feature)
            non_mono_feature = F.hardtanh(non_mono_feature, min_val=0.0, max_val=1.0)

        x = self.mono_fc_in(torch.cat([mono_feature, non_mono_feature], dim=1))
        x = F.relu(x)
        for i in range(int(len(self.mono_submods_out))):
            x = self.mono_submods_out[i](x)
            x = F.hardtanh(x, min_val=0.0, max_val=1.0)
            x = self.mono_dropouts[i](x)

            y = self.non_mono_submods_out[i](y)
            y = F.hardtanh(y, min_val=0.0, max_val=1.0)
            y = self.non_mono_dropouts[i](y)

            x = self.mono_submods_in[i](torch.cat([x, y], dim=1))
            x = F.relu(x)

            y = self.non_mono_submods_in[i](y)
            y = F.relu(y)

        x = self.mono_fc_last(x)

        y = self.non_mono_fc_last(y)

        out = x + y
        if self.normalize_regression:
            out = F.sigmoid(out)
        return out

    def reg_forward(self, feature_num, mono_num, bottleneck=10, num=512):
        in_list = []
        out_list = []
        if self.compress_non_mono:
            input_feature = torch.rand(num, mono_num + 10)
        else:
            input_feature = torch.rand(num, feature_num)
        input_mono = input_feature[:, :mono_num]
        input_non_mono = input_feature[:, mono_num:]
        input_mono.requires_grad = True
        x = self.mono_fc_in(torch.cat([input_mono, input_non_mono], dim=1))
        in_list.append(input_mono)

        x = F.relu(x)
        for i in range(int(len(self.mono_submods_out))):
            x = self.mono_submods_out[i](x)
            out_list.append(x)

            input_feature = torch.rand(num, 2 * bottleneck)
            input_mono = input_feature[:, :bottleneck]
            input_non_mono = input_feature[:, bottleneck:]
            in_list.append(input_mono)
            in_list[-1].requires_grad = True

            x = self.mono_submods_in[i](torch.cat([input_mono, input_non_mono], dim=1))
            x = F.relu(x)

        x = self.mono_fc_last(x)
        out_list.append(x)
        return in_list, out_list

    def decay_dropout(self, rate):
        # rate the dropout probability
        for i in range(len(self.mono_dropouts)):
            self.mono_dropouts[i].p = self.mono_dropouts[i].p * rate
        for i in range(len(self.non_mono_dropouts)):
            self.non_mono_dropouts[i].p = self.non_mono_dropouts[i].p * rate


def get_mvnn(args, input_shape,n_dummy=1):
    if not args.use_dummy:
        n_dummy = 0
    capacity_generic_goods = np.array([1 for _ in range(input_shape - n_dummy)])
    model = MVNN_GENERIC(input_dim=input_shape - n_dummy,
                         num_hidden_layers=mvnn_parameters['num_hidden_layers'],
                         num_hidden_units=mvnn_parameters['num_hidden_units'],
                         layer_type=mvnn_parameters['layer_type'],
                         target_max=mvnn_parameters['target_max'],
                         lin_skip_connection=args.lin_skip_connection,
                         dropout_prob=mvnn_parameters['dropout_prob'],
                         init_method=mvnn_parameters['init_method'],
                         random_ts=mvnn_parameters['random_ts'],
                         trainable_ts=mvnn_parameters['trainable_ts'],
                         init_E=mvnn_parameters['init_E'],
                         init_Var=mvnn_parameters['init_Var'],
                         init_b=mvnn_parameters['init_b'],
                         init_bias=mvnn_parameters['init_bias'],
                         init_little_const=mvnn_parameters['init_little_const'],
                         capacity_generic_goods=capacity_generic_goods,
                         output_size=1,
                         )
    return model

def get_mvnn_partial(args, input_shape, partial_mvnn_parameters):
    input_shape_mono = input_shape[0]
    input_shape_non_mono = input_shape[1]
    output_shape = input_shape[2]
    capacity_generic_goods = np.array([1 for _ in range(input_shape_mono)])
    capacity_generic_goods_final = np.array([1 for _ in range(args.output_inner_mvnn+args.non_mono_output_dim)])
    model = MVNN_GENERIC_PARTIAL(input_dim=input_shape_mono,
                                 num_hidden_layers=args.num_hidden_layers,
                                 num_hidden_units=args.num_hidden_units,
                                 dropout_prob=args.dropout_prob,
                                 layer_type=partial_mvnn_parameters['layer_type'],
                                 target_max=partial_mvnn_parameters['target_max'],
                                 init_method=partial_mvnn_parameters['init_method'],
                                 random_ts=partial_mvnn_parameters['random_ts'],
                                 trainable_ts=args.trainable_ts,
                                 init_E=partial_mvnn_parameters['init_E'],
                                 init_Var=partial_mvnn_parameters['init_Var'],
                                 init_b=partial_mvnn_parameters['init_b'],
                                 init_bias=partial_mvnn_parameters['init_bias'],
                                 init_little_const=partial_mvnn_parameters['init_little_const'],
                                 lin_skip_connection=args.lin_skip_connection,
                                 output_inner_mvnn = args.output_inner_mvnn,

                                 non_mono_input_dim=input_shape_non_mono,
                                 non_mono_num_hidden_layers=args.non_mono_num_hidden_layers,
                                 non_mono_num_hidden_units=args.non_mono_num_hidden_units,
                                 non_mono_output_dim=args.non_mono_output_dim,
                                 non_mono_lin_skip_connection=args.non_mono_lin_skip_connection,
                                 non_mono_dropout_prob=args.non_mono_dropout_prob,

                                 final_num_hidden_layers=args.final_num_hidden_layers,
                                 final_num_hidden_units=args.final_num_hidden_units,
                                 final_dropout_prob=args.final_dropout_prob,
                                 final_trainable_ts=args.final_trainable_ts,
                                 final_lin_skip_connection=args.final_lin_skip_connection,
                                 final_output_inner_mvnn = output_shape,
                                 #final_output_inner_mvnn = args.final_output_inner_mvnn,

                                 final_layer_type=partial_mvnn_parameters['layer_type'],
                                 final_target_max=partial_mvnn_parameters['target_max'],
                                 final_init_method=partial_mvnn_parameters['init_method'],
                                 final_random_ts=partial_mvnn_parameters['random_ts'],
                                 final_init_E=partial_mvnn_parameters['init_E'],
                                 final_init_Var=partial_mvnn_parameters['init_Var'],
                                 final_init_b=partial_mvnn_parameters['init_b'],
                                 final_init_bias=partial_mvnn_parameters['init_bias'],
                                 final_init_little_const=partial_mvnn_parameters['init_little_const'],
                                 final_capacity_generic_goods=capacity_generic_goods_final,
                                 capacity_generic_goods=capacity_generic_goods,
                                 )
    return model

def get_pumnn(args, pumnn_parameters, input_shape , device="cpu"):
    # this allows to set embedding network output size
    if pumnn_parameters['out_embedding_same'] == True :
        out_embedding = input_shape[1]
    else:
        out_embedding = pumnn_parameters['out_embedding']

    model = PartialEmbeddingNet(in_embedding=input_shape[1], # embedd the non_mono part
                                in_main=input_shape[0], # this is mono size
                                out_embedding=out_embedding,
                                device=device,
                                num_embedding_layers=pumnn_parameters['num_embedding_layers'],
                                num_embedding_hiddens=umnn_parameters['num_embedding_hiddens'],
                                num_main_hidden_layers=umnn_parameters['num_main_hidden_layers'],
                                num_main_hidden_nodes=umnn_parameters['num_main_hidden_nodes'],
                                dropout_prob = args.dropout_prob,
                                nb_steps=umnn_parameters['nb_steps'])
    return model
def get_umnn(umnn_parameters, input_shape, device="cpu"):
    model = EmbeddingNet(in_embedding=input_shape,
                         in_main=input_shape,
                         out_embedding=input_shape,
                         device=device,
                         num_embedding_layers=umnn_parameters['num_embedding_layers'],
                         num_embedding_hiddens=umnn_parameters['num_embedding_hiddens'],
                         num_main_hidden_layers=umnn_parameters['num_main_hidden_layers'],
                         num_main_hidden_nodes=umnn_parameters['num_main_hidden_nodes'],
                         nb_steps=umnn_parameters['nb_steps'])
    return model


pcert_parameters = {"output_parameters": 1, "mono_sub_num" : 2, "non_mono_sub_num": 2, "mono_hidden_num": 20, "non_mono_hidden_num": 10, "compress_mono": False, "compress_non_mono": False, "normalize_regression": False}

def get_cert(args, train_shape, cert_parameters):
    mono_shape = train_shape[0]
    non_mono_shape = train_shape[1]
    if args.model == "CERT":
        print( "Normal CERT should not be run from this file !!" ) 
        model = MLP_relu(train_shape[0] -1, 1,1,1,5,5,False, False)
    return model

def get_pcert(args, train_shape, cert_parameters):
    print("Train shapes while loading", train_shape)
    print(" args while loading" , args) 
    model = MLP_relu(mono_feature= train_shape[0],
                     non_mono_feature= train_shape[1],
                     mono_sub_num=args.num_hidden_layers,
                     non_mono_sub_num=args.num_hidden_layers,

                     mono_hidden_num=args.num_hidden_units,
                     non_mono_hidden_num=args.num_hidden_units,

                     compress_non_mono=args.compress_non_mono,
                     normalize_regression=args.normalize_regression,
                     dropout_prob=args.dropout_prob,
                     bottleneck=args.bottleneck
                     )

    return model

def get_mono_minmax(args, input_shape):
    # hard code for test
    model = MonotoneMinMax(
        mono_mode=              args.mono_mode,
        mono_in_features=       input_shape[0],
        mono_num_groups=        args.num_groups,
        mono_group_size=        args.group_size,
        non_mono_in_features=   input_shape[1],
        non_mono_num_groups=    args.non_mono_num_groups,
        non_mono_group_size=    args.non_mono_group_size,
        dropout_prob=           args.dropout_prob,
        )
    return model
def get_minmax(args, input_shape):
    # hard code for test
    model = MinMax(in_features= input_shape - 1,
                   num_groups=args.num_groups,
                   group_size=args.group_size,
                   )
    return model

# TODOS:
# define nn parameters -> use default for now (see mvnn_generic.py)
# define epochs and test size
# choose loss function -> use MSE for now
# set training options -> write script
# define optimizer + params
# init w and biases
# log metrics
# store model
# generate plots -> separate File
def train_model(args, model, train, val, test,  metrics,  bidder_id=1, cumm_batch=0, cumm_epoch=0, seed=100, infos=[0,0]):
    print("--Starting Training--")
    train_shape = [train[0][0].shape[0], train[0][1].shape[0]]
    print(" train shape is : ", train_shape) 
    # metrics for regression
    loss_mse = torch.nn.MSELoss()

    ### define metrics ###
    loss_mae = torch.nn.L1Loss()
    loss_evar = skm.explained_variance_score
    loss_medabs = skm.median_absolute_error
    loss_r2 = skm.r2_score
    loss_maxerr = skm.max_error
    loss_mape = skm.mean_absolute_percentage_error
    loss_d2tw = skm.d2_tweedie_score
    loss_mpl = skm.mean_pinball_loss
    loss_d2pl = skm.d2_pinball_score
    loss_d2abserr = skm.d2_absolute_error_score
    loss_kendall = kendalltau

    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)

    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay= args.l2_rate)

    wandb.watch(model, log="all")
    # this is only relevant for CERT where we add previous batch iterations
    batch_num = 0 
    epoch_num = 0

    curr_dropout = args.dropout_prob
    if infos[2] == 1:
        batch_num = infos[0] 
        epoch_num = infos[1]
        
    seed_metrics_train = []
    seed_metrics_val = []
    seed_metrics_test = []
    for e in tqdm(range(args.epochs)):
        epoch_num += 1
        for batch in train_loader:
            #batch = batch.to(device)
            batch_num +=1
            reg_loss = 0
            cert_loss = 0
            print(batch[0].shape,  "/ is batch shape : ", batch[1].shape, batch[2].shape) 

            optimizer.zero_grad()
            if args.model == 'CERT':
                n_dummy = 1
                predictions = model.forward(batch[0][:,:-n_dummy], batch[0][:,-n_dummy:])
                cert_loss = loss_mse(predictions.squeeze(1),batch[1][:,bidder_id])
                in_list, out_list = model.reg_forward(train_shape, train_shape - n_dummy)
                reg_loss = generate_regularizer(in_list, out_list)
                loss_tot = cert_loss + model.lam * reg_loss
                loss_tot.backward()
                optimizer.step()

            if args.model == 'PCERT':
                print(" batches are: " , batch[0], " and " , batch[1])
                predictions = model.forward(batch[0],batch[1])
                cert_loss = loss_mse(predictions.squeeze(1), batch[2].squeeze(1))
                in_list, out_list = model.reg_forward(train_shape[1] + train_shape[0], train_shape[0])
                reg_loss = generate_regularizer(in_list, out_list)
                loss_tot = cert_loss + model.lam * reg_loss
                loss_tot.backward()
                optimizer.step()


            elif args.model == 'UMNN':
                model.set_steps(int(torch.randint(1,10, [1])))
                predictions = model.forward(batch[0])
                loss_tot = loss_mse(predictions.squeeze(1),batch[1][:,bidder_id])
                loss_tot.backward()
                optimizer.step()

            elif args.model == 'PUMNN':
                model.set_steps(int(torch.randint(1,10, [1])))
                predictions = model.forward(batch[0], batch[1])
                loss_tot = loss_mse(predictions.squeeze(1),batch[2].squeeze(1))
                loss_tot.backward()
                optimizer.step()
                

            elif args.model == "MVNN":
                predictions = model.forward(batch[0])
                loss_tot = loss_mse(predictions.squeeze(1),batch[1][:,bidder_id])
                loss_tot.backward()
                optimizer.step()
                model.transform_weights()

            elif args.model == "PMVNN":
                predictions = model.forward(batch[0], batch[1])
                loss_tot = loss_mse(predictions.squeeze(1),batch[2].squeeze(1))
                loss_tot.backward()
                optimizer.step()
                model.transform_weights()

            elif args.model == "MINMAX" or args.model == "MONOMINMAX":
                predictions = model.forward(batch[0],batch[1])
                loss_tot = loss_mse(predictions.squeeze(1),batch[2].squeeze())
                loss_tot.backward()
                optimizer.step()
                if args.model == "MONOMINMAX":
                    model.set_weights()

            model.decay_dropout(rate=args.dropout_decay)
            curr_dropout = curr_dropout*args.dropout_decay

            if args.model =="CERT" or args.model == "PCERT":
                if args.dataset == "blog" or args.dataset == "compas":
                    print("logging data for pcert and new data cert is ", cert_loss)
                    seed_metrics_train.append([loss_tot.item(),
                                               loss_mae(predictions.squeeze(1), batch[2].squeeze(1)).item(),
                                               loss_mse(predictions.squeeze(1), batch[2].squeeze(1)).item(),
                                               loss_evar(y_true=batch[2].squeeze(1),
                                                         y_pred=predictions.squeeze(1).detach()).item(),
                                               loss_medabs(y_true=batch[2].squeeze(1),
                                                           y_pred=predictions.squeeze(1).detach()).item(),
                                               loss_r2(y_true=batch[2].squeeze(1),
                                                       y_pred=predictions.squeeze(1).detach()).item(),
                                               loss_maxerr(y_true=batch[2].squeeze(1),
                                                           y_pred=predictions.squeeze(1).detach()).item(),
                                               loss_mape(y_true=batch[2].squeeze(1),
                                                         y_pred=predictions.squeeze(1).detach()).item(),
                                               loss_d2tw(y_true=batch[2].squeeze(1),
                                                         y_pred=predictions.squeeze(1).detach()).item(),
                                               loss_mpl(y_true=batch[2].squeeze(1),
                                                        y_pred=predictions.squeeze(1).detach()).item(),
                                               loss_d2pl(y_true=batch[2].squeeze(1),
                                                         y_pred=predictions.squeeze(1).detach()).item(),
                                               loss_d2abserr(y_true=batch[2].squeeze(1),
                                                             y_pred=predictions.squeeze(1).detach()).item(),
                                               kendalltau(batch[2].squeeze(1), predictions.squeeze(1).detach())[0],
                                               kendalltau(batch[2].squeeze(1), predictions.squeeze(1).detach())[1],
                                               batch_num,
                                               epoch_num,
                                               #numpy.array(reg_loss.item()),
                                               #reg_loss.item(),
                                               reg_loss.detach().item(),
                                               cert_loss.detach().item(),
                                               curr_dropout,
                                               ])
                else:
                    seed_metrics_train.append([loss_tot.item(),
                                               loss_mae(predictions.squeeze(1), batch[1][:, bidder_id]).item(),
                                               loss_mse(predictions.squeeze(1), batch[1][:, bidder_id]).item(),
                                               loss_evar(y_true=batch[1][:, bidder_id],
                                                         y_pred=predictions.squeeze(1).detach()).item(),
                                               loss_medabs(y_true=batch[1][:, bidder_id],
                                                           y_pred=predictions.squeeze(1).detach()).item(),
                                               loss_r2(y_true=batch[1][:, bidder_id],
                                                       y_pred=predictions.squeeze(1).detach()).item(),
                                               loss_maxerr(y_true=batch[1][:, bidder_id],
                                                           y_pred=predictions.squeeze(1).detach()).item(),
                                               loss_mape(y_true=batch[1][:, bidder_id],
                                                         y_pred=predictions.squeeze(1).detach()).item(),
                                               loss_d2tw(y_true=batch[1][:, bidder_id],
                                                         y_pred=predictions.squeeze(1).detach()).item(),
                                               loss_mpl(y_true=batch[1][:, bidder_id],
                                                        y_pred=predictions.squeeze(1).detach()).item(),
                                               loss_d2pl(y_true=batch[1][:, bidder_id],
                                                         y_pred=predictions.squeeze(1).detach()).item(),
                                               loss_d2abserr(y_true=batch[1][:, bidder_id],
                                                             y_pred=predictions.squeeze(1).detach()).item(),
                                               kendalltau(batch[1][:, bidder_id], predictions.squeeze(1).detach())[0],
                                               kendalltau(batch[1][:, bidder_id], predictions.squeeze(1).detach())[1],
                                               batch_num,
                                               epoch_num,
                                               numpy.array(reg_loss.detach()),
                                               numpy.array(cert_loss.detach()),
                                               curr_dropout,
                                               ])
            else:
                if args.dataset == "blog" or args.dataset =="compas":
                    seed_metrics_train.append([loss_tot.item(),
                                               loss_mae(predictions.squeeze(1), batch[2].squeeze(1)).item(),
                                               loss_mse(predictions.squeeze(1), batch[2].squeeze(1)).item(),
                                               loss_evar(y_true=batch[2].squeeze(1), y_pred=predictions.squeeze(1).detach()).item(),
                                               loss_medabs(y_true=batch[2].squeeze(1), y_pred=predictions.squeeze(1).detach()).item(),
                                               loss_r2(y_true=batch[2].squeeze(1), y_pred=predictions.squeeze(1).detach()).item(),
                                               loss_maxerr(y_true=batch[2].squeeze(1), y_pred=predictions.squeeze(1).detach()).item(),
                                               loss_mape(y_true=batch[2].squeeze(1), y_pred=predictions.squeeze(1).detach()).item(),
                                               loss_d2tw(y_true=batch[2].squeeze(1), y_pred=predictions.squeeze(1).detach()).item(),
                                               loss_mpl(y_true=batch[2].squeeze(1), y_pred=predictions.squeeze(1).detach()).item(),
                                               loss_d2pl(y_true=batch[2].squeeze(1), y_pred=predictions.squeeze(1).detach()).item(),
                                               loss_d2abserr(y_true=batch[2].squeeze(1), y_pred=predictions.squeeze(1).detach()).item(),
                                               kendalltau(batch[2].squeeze(1), predictions.squeeze(1).detach())[0],
                                               kendalltau(batch[2].squeeze(1), predictions.squeeze(1).detach())[1],
                                               batch_num,
                                               epoch_num,
                                               reg_loss,
                                               cert_loss,
                                               curr_dropout,
                                               ])
                else:
                    seed_metrics_train.append([loss_tot.item(),
                                               loss_mae(predictions.squeeze(1), batch[1][:, bidder_id]).item(),
                                               loss_mse(predictions.squeeze(1), batch[1][:, bidder_id]).item(),
                                               loss_evar(y_true=batch[1][:, bidder_id], y_pred=predictions.squeeze(1).detach()).item(),
                                               loss_medabs(y_true=batch[1][:, bidder_id], y_pred=predictions.squeeze(1).detach()).item(),
                                               loss_r2(y_true=batch[1][:, bidder_id], y_pred=predictions.squeeze(1).detach()).item(),
                                               loss_maxerr(y_true=batch[1][:, bidder_id], y_pred=predictions.squeeze(1).detach()).item(),
                                               loss_mape(y_true=batch[1][:, bidder_id], y_pred=predictions.squeeze(1).detach()).item(),
                                               loss_d2tw(y_true=batch[1][:, bidder_id], y_pred=predictions.squeeze(1).detach()).item(),
                                               loss_mpl(y_true=batch[1][:, bidder_id], y_pred=predictions.squeeze(1).detach()).item(),
                                               loss_d2pl(y_true=batch[1][:, bidder_id], y_pred=predictions.squeeze(1).detach()).item(),
                                               loss_d2abserr(y_true=batch[1][:, bidder_id], y_pred=predictions.squeeze(1).detach()).item(),
                                               kendalltau(batch[1][:, bidder_id], predictions.squeeze(1).detach())[0],
                                               kendalltau(batch[1][:, bidder_id], predictions.squeeze(1).detach())[1],
                                               batch_num,
                                               epoch_num,
                                               reg_loss,
                                               cert_loss,
                                               curr_dropout,
                                               ])

        ### Validation ###
        print("START validation")
        val_size = len(val)
        val_loader = torch.utils.data.DataLoader(val, batch_size=val_size, shuffle=True)
        for batch in tqdm(val_loader):
            if args.model == "MVNN" or args.model == "UMNN":
                predictions = model.forward(batch[0])
            elif args.model == "CERT":
                predictions = model.forward(batch[0][:, :-n_dummy], batch[0][:, -n_dummy:])
            elif args.model == "PMVNN":
                predictions = model.forward(batch[0], batch[1])
            elif args.model == "PCERT":
                predictions = model.forward(batch[0], batch[1])
            elif args.model == "PUMNN":
                model.set_steps(int(torch.randint(1,10, [1])))
                predictions = model.forward(batch[0], batch[1])
            elif args.model == "MINMAX" or args.model == "MONOMINMAX":
                predictions = model.forward(batch[0], batch[1])

            if args.dataset == "blog" or args.dataset == "compas":
                print("validating blog ")
                val_loss = loss_mse(predictions.squeeze(1), batch[2].squeeze(1))
                seed_metrics_val.append([val_loss.item(),
                                         loss_mae(predictions.squeeze(1), batch[2].squeeze(1)).item(),
                                         loss_mse(predictions.squeeze(1), batch[2].squeeze(1)).item(),
                                         loss_evar(y_true=batch[2].squeeze(1), y_pred=predictions.squeeze(1).detach()).item(),
                                         loss_medabs(y_true=batch[2].squeeze(1), y_pred=predictions.squeeze(1).detach()).item(),
                                         loss_r2(y_true=batch[2].squeeze(1), y_pred=predictions.squeeze(1).detach()).item(),
                                         loss_maxerr(y_true=batch[2].squeeze(1), y_pred=predictions.squeeze(1).detach()).item(),
                                         loss_mape(y_true=batch[2].squeeze(1), y_pred=predictions.squeeze(1).detach()).item(),
                                         loss_d2tw(y_true=batch[2].squeeze(1), y_pred=predictions.squeeze(1).detach()).item(),
                                         loss_mpl(y_true=batch[2].squeeze(1), y_pred=predictions.squeeze(1).detach()).item(),
                                         loss_d2pl(y_true=batch[2].squeeze(1), y_pred=predictions.squeeze(1).detach()).item(),
                                         loss_d2abserr(y_true=batch[2].squeeze(1), y_pred=predictions.squeeze(1).detach()).item(),
                                         kendalltau(batch[2].squeeze(1), predictions.squeeze(1).detach())[0],
                                         kendalltau(batch[2].squeeze(1), predictions.squeeze(1).detach())[1],
                                         batch_num,
                                         epoch_num,
                                         ])
            else:
                val_loss = loss_mse(predictions.squeeze(1), batch[1][:, bidder_id])
                seed_metrics_val.append([val_loss.item(),
                                         loss_mae(predictions.squeeze(1), batch[1][:, bidder_id]).item(),
                                         loss_mse(predictions.squeeze(1), batch[1][:, bidder_id]).item(),
                                         loss_evar(y_true=batch[1][:, bidder_id], y_pred=predictions.squeeze(1).detach()).item(),
                                         loss_medabs(y_true=batch[1][:, bidder_id], y_pred=predictions.squeeze(1).detach()).item(),
                                         loss_r2(y_true=batch[1][:, bidder_id], y_pred=predictions.squeeze(1).detach()).item(),
                                         loss_maxerr(y_true=batch[1][:, bidder_id], y_pred=predictions.squeeze(1).detach()).item(),
                                         loss_mape(y_true=batch[1][:, bidder_id], y_pred=predictions.squeeze(1).detach()).item(),
                                         loss_d2tw(y_true=batch[1][:, bidder_id], y_pred=predictions.squeeze(1).detach()).item(),
                                         loss_mpl(y_true=batch[1][:, bidder_id], y_pred=predictions.squeeze(1).detach()).item(),
                                         loss_d2pl(y_true=batch[1][:, bidder_id], y_pred=predictions.squeeze(1).detach()).item(),
                                         loss_d2abserr(y_true=batch[1][:, bidder_id], y_pred=predictions.squeeze(1).detach()).item(),
                                         kendalltau(batch[1][:, bidder_id], predictions.squeeze(1).detach())[0],
                                         kendalltau(batch[1][:, bidder_id], predictions.squeeze(1).detach())[1],
                                         batch_num,
                                         epoch_num,
                                         ])
        print("END validation")
    print("Start Testing")
        ### Test ###
    test_size = len(test)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_size, shuffle=True)
    for batch in test_loader:
        #batch = batch.to(device)
        if args.model == "MVNN" or args.model == "UMNN":
            predictions = model.forward(batch[0])
        elif args.model == "CERT":
            predictions = model.forward(batch[0][:, :-n_dummy], batch[0][:, -n_dummy:])
        elif args.model == "PMVNN":
            predictions = model.forward(batch[0], batch[1])
        elif args.model == "PCERT":
            predictions = model.forward(batch[0], batch[1])
        elif args.model == "PUMNN":
            predictions = model.forward(batch[0], batch[1])
        elif args.model == "MINMAX" or args.model == "MONOMINMAX":
            predictions = model.forward(batch[0], batch[1])
        if args.dataset == "blog" or args.dataset =="compas":
            test_loss_tot = loss_mse(predictions.squeeze(1), batch[2].squeeze(1))
            print("testing blog ")
            seed_metrics_test.append([test_loss_tot.item(),
                                      loss_mae(predictions.squeeze(1), batch[2].squeeze(1)).item(),
                                      loss_mse(predictions.squeeze(1), batch[2].squeeze(1)).item(),
                                      loss_evar(y_true=batch[2].squeeze(1), y_pred=predictions.squeeze(1).detach()).item(),
                                      loss_medabs(y_true=batch[2].squeeze(1), y_pred=predictions.squeeze(1).detach()).item(),
                                      loss_r2(y_true=batch[2].squeeze(1), y_pred=predictions.squeeze(1).detach()).item(),
                                      loss_maxerr(y_true=batch[2].squeeze(1), y_pred=predictions.squeeze(1).detach()).item(),
                                      loss_mape(y_true=batch[2].squeeze(1), y_pred=predictions.squeeze(1).detach()).item(),
                                      loss_d2tw(y_true=batch[2].squeeze(1), y_pred=predictions.squeeze(1).detach()).item(),
                                      loss_mpl(y_true=batch[2].squeeze(1), y_pred=predictions.squeeze(1).detach()).item(),
                                      loss_d2pl(y_true=batch[2].squeeze(1), y_pred=predictions.squeeze(1).detach()).item(),
                                      loss_d2abserr(y_true=batch[2].squeeze(1), y_pred=predictions.squeeze(1).detach()).item(),
                                      kendalltau(batch[2].squeeze(1), predictions.squeeze(1).detach())[0],
                                      kendalltau(batch[2].squeeze(1), predictions.squeeze(1).detach())[1],
                                      batch_num,
                                      epoch_num,
                                      ])
        else: 
            test_loss_tot = loss_mse(predictions.squeeze(1), batch[1][:, bidder_id])
            seed_metrics_test.append([test_loss_tot.item(),
                                      loss_mae(predictions.squeeze(1), batch[1][:, bidder_id]).item(),
                                      loss_mse(predictions.squeeze(1), batch[1][:, bidder_id]).item(),
                                      loss_evar(y_true=batch[1][:, bidder_id], y_pred=predictions.squeeze(1).detach()).item(),
                                      loss_medabs(y_true=batch[1][:, bidder_id], y_pred=predictions.squeeze(1).detach()).item(),
                                      loss_r2(y_true=batch[1][:, bidder_id], y_pred=predictions.squeeze(1).detach()).item(),
                                      loss_maxerr(y_true=batch[1][:, bidder_id], y_pred=predictions.squeeze(1).detach()).item(),
                                      loss_mape(y_true=batch[1][:, bidder_id], y_pred=predictions.squeeze(1).detach()).item(),
                                      loss_d2tw(y_true=batch[1][:, bidder_id], y_pred=predictions.squeeze(1).detach()).item(),
                                      loss_mpl(y_true=batch[1][:, bidder_id], y_pred=predictions.squeeze(1).detach()).item(),
                                      loss_d2pl(y_true=batch[1][:, bidder_id], y_pred=predictions.squeeze(1).detach()).item(),
                                      loss_d2abserr(y_true=batch[1][:, bidder_id], y_pred=predictions.squeeze(1).detach()).item(),
                                      kendalltau(batch[1][:, bidder_id], predictions.squeeze(1).detach())[0],
                                      kendalltau(batch[1][:, bidder_id], predictions.squeeze(1).detach())[1],
                                      batch_num,
                                      epoch_num,
                                      ])

    metrics[0].append(seed_metrics_train)
    metrics[1].append(seed_metrics_val)
    metrics[2].append(seed_metrics_test)
    infos[0] = batch_num
    infos[1] = epoch_num
    return model, metrics, infos

#TODO Work on defining parameters smoothly
def get_model(args, train_shape):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device is : ", device)
    ### define model ###
    if args.model == 'MVNN':
        model = get_mvnn(args, train_shape)

    elif args.model == 'UMNN':
        model = get_umnn(umnn_parameters, train_shape, device)

    elif args.model == 'CERT':
        print("LOADING CERT")
        model = get_cert(args, train_shape, cert_parameters)

    elif args.model == 'PCERT':
        print("LOADING PCERT")
        model = get_pcert(args, train_shape, cert_parameters)

    elif args.model == "PMVNN":
        print("LOADING PMVNN")
        model = get_mvnn_partial(args, train_shape, pmvnn_parameters)

    elif args.model== "PUMNN":
        print("LOADING PUMNN")
        model = get_pumnn(args, pumnn_parameters,train_shape, device)

    elif args.model == 'MINMAX':
        print("LOADING MINMAX")
        model = get_minmax(args, train_shape)

    elif args.model == 'MONOMINMAX':
        print("LOADING MONOMINMAX")
        model = get_mono_minmax(args, train_shape)
    else: 
        print( "ERROR WRONG MODEL: ", args.model)
        model = None
        

    return model

def log_metrics(args, metrics):
    print("Logging metrics") 
    mets_train = numpy.array([numpy.array(xi) for xi in metrics[0]])
    mets_val = numpy.array([numpy.array(xi) for xi in metrics[1]])
    mets_test = numpy.array([numpy.array(xi) for xi in metrics[2]])
    dims_train = mets_train.shape
    dims_val = mets_val.shape
    dims_test = mets_test.shape

    mean_train = np.mean(mets_train ,axis= 0) 
    mean_val = np.mean(mets_val ,axis= 0) 
    mean_test = np.mean(mets_test ,axis= 0) 

    print("Lengths are: ", dims_train[1], dims_val[1], dims_test[1])

    for i in range(dims_train[1]):
        wandb.log({"loss_tot": mean_train[i,0],
                   "loss_mae": mean_train[i,1],
                   "loss_mse": mean_train[i,2],
                   "loss_evar": mean_train[i,3],
                   "loss_medabs": mean_train[i,4],
                   "loss_r2": mean_train[i,5],
                   "loss_maxerr": mean_train[i,6],
                   "loss_mape": mean_train[i,7],
                   "loss_d2tw": mean_train[i,8],
                   "loss_mpl": mean_train[i,9],
                   "loss_d2pl": mean_train[i,10],
                   "loss_d2abserr": mean_train[i,11],
                   "kendall_tau_statistics": mean_train[i,12],
                   "kendall_tau_p_val": mean_train[i,13],
                   "Batch_num_train": mean_train[i,14],
                   "Epoch_train": mean_train[i,15],
                   "Cert_regularizer": mean_train[i, 16],
                   "Cert_loss": mean_train[i, 17],
                   "Dropout": mean_train[i, 18],
                   })
    for i in range(dims_val[1]):
        wandb.log({"val_loss_tot": mean_val[i,0],
                   "val_loss_mae": mean_val[i,1],
                   "val_loss_mse": mean_val[i,2],
                   "val_loss_evar": mean_val[i,3],
                   "val_loss_medabs": mean_val[i,4],
                   "val_loss_r2": mean_val[i,5],
                   "val_loss_maxerr": mean_val[i,6],
                   "val_loss_mape": mean_val[i,7],
                   "val_loss_d2tw": mean_val[i,8],
                   "val_loss_mpl": mean_val[i,9],
                   "val_loss_d2pl": mean_val[i,10],
                   "val_loss_d2abserr": mean_val[i,11],
                   "val_kendall_tau_statistics": mean_val[i,12],
                   "val_kendall_tau_p_val": mean_val[i,13],
                   "Batch_num_val": mean_val[i,14],
                   "Epoch_val": mean_val[i,15]})
    for i in range(dims_test[1]):
        wandb.log({"test_loss_tot": mean_test[i,0],
                   "test_loss_mae": mean_test[i,1],
                   "test_loss_mse": mean_test[i,2],
                   "test_loss_evar": mean_test[i,3],
                   "test_loss_medabs": mean_test[i,4],
                   "test_loss_r2": mean_test[i,5],
                   "test_loss_maxerr": mean_test[i,6],
                   "test_loss_mape": mean_test[i,7],
                   "test_loss_d2tw": mean_test[i,8],
                   "test_loss_mpl": mean_test[i,9],
                   "test_loss_d2pl": mean_test[i,10],
                   "test_loss_d2abserr": mean_test[i,11],
                   "test_kendall_tau_statistics": mean_test[i,12],
                   "test_kendall_tau_p_val": mean_test[i,13],
                   "Batch_num_test": mean_test[i,14],
                   "Epoch_test": mean_test[i,15]})

def main(args=None):

    print("--Starting main--")
    parser = init_parser()
    args = parser.parse_args()

    ## initialise wandb
    run = wandb.init()
    #wandb.log({"Started": True})

    args.__dict__.update(wandb.config)

    wandb.config.update(args, allow_val_change=True)
    print(wandb.config ) 

    # log  parameters
    group_id = str(args.model) + str(args.dataset) + str(args.bidder_id)
    run_id = group_id  + "Rand_id_"+ str(np.random.randint(300))

    run.name = run_id
    if args.model == "MVNN":
        wandb.config.update(mvnn_parameters, allow_val_change=True)

    elif args.model == "UMNN":    
        wandb.config.update(umnn_parameters, allow_val_change=True)

    elif args.model == "CERT":
        wandb.config.update(cert_parameters, allow_val_change=True)

    elif args.model == "PMVNN":
        wandb.config.update(pmvnn_parameters, allow_val_change=True)

    elif args.model == "PUMNN":
        wandb.config.update(pumnn_parameters, allow_val_change=True)

    elif args.model == "PCERT":
        wandb.config.update(pcert_parameters, allow_val_change=True)

    elif args.model == "MINMAX":
        wandb.config.update(minmax_parameters, allow_val_change=True)

    elif args.model == "MONOMINMAX":
        wandb.config.update(minmax_parameters, allow_val_change=True)


    # define metrics
    metrics = [[],[],[]]
    infos = [0,0,0] # cumulative batch and epoch last is bool if CERT is extended  


    # TODO remove bidder id loop 
    #bidder_ids = [args.bidder_id]
    #for bidder in bidder_ids:
    for num, seed in enumerate(range(args.initial_seed, args.initial_seed + args.num_seeds)):
        wandb.log({"model": args.model, "dataset": args.dataset})
        print( " Running model: ", args.model, " and dataset: ", args.dataset)


        ### load dataset ###
        train, val, test = load_dataset(args, train_percent=args.train_percent,seed=seed)
        train_shape = [train[0][0].shape[0], train[0][1].shape[0], train[0][2].shape[0]]

        print(train_shape, " is the train shape and seed is ", seed)
        print(train[0], " is the full train shape and seed is ", seed)
        print("--- Loaded dataset successfully ---")

        ### define model ###
        model = get_model(args, train_shape)
        print("--- Loaded model successfully ---")


        ### train model ###
        model, metrics, infos = train_model(args, model, train, val, test,  metrics, bidder_id=args.bidder_id, seed=seed, infos=infos)
        print("--- Trained model successfully ---")

    log_metrics(args, metrics)
    wandb.finish()
    """
        if args.model == "CERT":
            pass
            #    n_dummy = 1
            #mono_flag = certify_neural_network(model, train_shape-n_dummy)
            while not mono_flag:
                model, metrics, infos = train_model(args, model, train, val, test, metrics, bidder_id=bidder, seed=seed, infos=infos)
                assert(args.use_dummy)
                mono_flag = certify_neural_network(model, train_shape-n_dummy)
                if not mono_flag:
                    model.lam *= 10
                    print("Network not monotonic, increasing regularization strength to ", model.lam)
                    wandb.log({"lam":model.lam, "train_batch": infos[0]})

                    if model.lam == 1000:
                        print("Exiting because of too many trys in CERT")
                        mono_flag = True
           """

def start_agent(inputs):
    print("Starting Agent!")
    sweep_id = inputs[0]
    count = inputs[1]
    wandb.agent(sweep_id, function=main, count=count)
    print("Agent Done")
    return 0
if __name__ == "__main__":
    print("--Starting wandb sweep init-- ")

    #os.environ['WANDB_SILENT'] = "true"
    os.environ['WANDB_MODE'] = "offline"
    #os.environ['WANDB_DIR'] = os.path.abspath("/cluster/scratch/filles/")
    #os.chdir('/cluster/scratch/filles')
    print(os.getcwd(), " : is the current working directory")
 
    #parser = init_parser()
    #args = parser.parse_args()
    #group_id = str(args.model) + str(args.dataset) + str(args.bidder_id)
    #os.environ["WANDB_RUN_GROUP"] = "experiment-" + group_id 
    #MODEL = "CERT"
    #MODEL = "PMVNN"
    MODEL = "PCERT"
    #MODEL = "PUMNN"
    #MODEL = "MINMAX"
    #MODEL = "MONOMINMAX"
    print("Running model: ", MODEL)

    sweep_config = { 
        "method": "random", 
        "metric": {"goal": "minimize", "name": "val_loss_tot"}, 
        "parameters": {
            # general Params 
            "model": {"values":[str(MODEL)]},
            #"dataset": {"values":["compas"]}, 
            "dataset": {"values":["blog"]}, 
            "bidder_id":{ "values": [0]},
            "num_train_points":{ "values": [100]},


            #"epochs":{ "values": [10]},
            "epochs":{ "values": [100, 200, 400]},
            "learning_rate": {"values": [ 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05 ]},
            "batch_size": { "values": [10, 50]},
            "l2_rate": { "values": [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0]},
            "dropout_prob": {"values": [0., 0.1, 0.2, 0.3, 0.4 ,0.5]},
            #"dropout_prob": {"values": [0.]},
            
            # PMVNN Params
            #"final_num_hidden_layers": { "values" : [1,2,3,4]},
            #"final_num_hidden_units": { "values": [16,32,64,128,256]},
            #"final_dropout_prob": {"values": [0., 0.1, 0.2, 0.3, 0.4 ,0.5]},
            #"final_trainable_ts": {"values": ["True", "False"]},
            #"final_lin_skip_connection": {"values": ["True", "False"]},

            #"non_mono_dropout_prob": {"values": [0., 0.1, 0.2, 0.3, 0.4 ,0.5]},
            #"non_mono_lin_skip_connection": {"values": ["True", "False"]},

            #"output_inner_mvnn": { "values": [8,16,32,64]},
            #"non_mono_output_dim": { "values": [8,16,32,64]},
            #"non_mono_output_dim": { "values": [8,16,32,64]},
            #"final_output_inner_mvnn": { "values": [8,16,32,64]},

            ###CERT Params
            "compress_non_mono": {"values": ["True", "False"]},
            "normalize_regression": {"values": ["True", "False"]},
            "bottleneck":{"values": [10]},


            ### PMVNN and PCERT Params
            "num_hidden_layers": { "values" : [1,2,3,4]},
            "num_hidden_units": { "values": [16,32,64,128,256]},
            "non_mono_num_hidden_units": { "values": [16,32,64,128,256]},
            #"non_mono_num_hidden_layers": { "values" : [1,2,3,4]},


            #MINMAX Params
            #"num_groups": {"values": [32, 64, 128, 256]},
            #"group_size": {"values": [8, 16, 32, 64, 128]},
            #"non_mono_num_groups": {"values": [32, 64, 128, 256]},
            #"non_mono_group_size": {"values": [8, 16, 32, 64, 128]},
            #MONOMINMAX Params
            #"mono_mode": { "values": ["x2","weights"]},
        
            },
        }

    sweep_id = wandb.sweep(sweep=sweep_config, project="Experiment 2 HPO blog 26 02   ")
    count = 10
    num_threads = 12
    with mp.Pool(num_threads) as p :
       p.map(start_agent,[[sweep_id,count] for _ in range(num_threads)])

    #parser = init_parser()
    #args = parser.parse_args()
    #main(args)

