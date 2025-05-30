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
#from mvnns.mvnn_generic_mixed import MVNN_GENERIC_MIXED



def init_parser():
    parser = argparse.ArgumentParser()

    ### experiment parameters ###
    parser.add_argument("--dataset", help="dataset to use", default="lsvm", choices=['gsvm' , 'lsvm', 'srvm', 'mrvm', 'blog'] )
    parser.add_argument("--nbids", help="number of bids to use", default=25000)
    parser.add_argument("--bidder_id", help="bidder id to use", default=0)
    parser.add_argument('-m','--model',  type=str, help='Choose model to train: UMNN, MVNN', choices=['UMNN','MVNN','CERT', "MMVNN"], default='MVNN')

    parser.add_argument("-tp","--train_percent", type=float, default=0.0, help="percentage of data to use for training")
    parser.add_argument("-ud","--use_dummy", type=bool, default=True, help="use dummy dataset")
    parser.add_argument("-ns","--num_seeds", type=int, default=1, help="number of seeds to use for hpo")
    parser.add_argument("-is","--initial_seed", type=int, default=100, help="initial seed to use for hpo")

    ### training parameters ###
    parser.add_argument("-e","--epochs", help="number of epochs to train", default=1)
    parser.add_argument("--batch_size", help="batch size to use", default=128)
    parser.add_argument("--learning_rate", help="learning rate", default=0.001)
    #parser.add_argument("--loss", help="ltenary operator expression c++oss function to use", default="mse")
    #parser.add_argument("--optimizer", help="optimizer to use", default="adam")
    parser.add_argument("--l2_rate", help="l2 norm", default=0.)

    ### model parameters ###
    parser.add_argument("--num_hidden_layers", help="number of hidden layers", default=12)
    parser.add_argument("--num_hidden_units", help="number of hidden units", default=23)
    parser.add_argument("--layer_type", help="layer type", default="MVNNLayerReLUProjected")
    parser.add_argument("--target_max", help="target max", default=1)
    parser.add_argument("--lin_skip_connection", type=bool,  help="linear skip connection", default=False)
    parser.add_argument("--final_lin_skip_connection", type=bool,  help="linear skip connection", default=False)
    parser.add_argument("--dropout_prob", help="dropout probability", default=0)
    parser.add_argument("--scale", help="scale to 0-1", type= bool, default=True)
    # for CERT
    parser.add_argument("--compress_non_mono", help="compressing mono inputs", type= bool, default=False)
    parser.add_argument("--normalize_regression", help="normalizing regression", type= bool, default=False)
    #parser.add_argument("--init_method", help="initialization method", default="custom")
    #parser.add_argument("--random_ts", help="random ts", default=[0,1])
    #parser.add_argument("--trainable_ts", help="trainable ts", default=True)
    #parser.add_argument("--init_E", help="init E", default=1)
    #parser.add_argument("--init_Var", help="init Var", default=0.09)
    #parser.add_argument("--init_b", help="init b", default=0.05)
    #parser.add_argument("--init_bias", help="init bias", default=0.05)

    return parser

#TODO: take this from a file
### default parameters ###
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
"""
final_input_dim: int,
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
                 final_output_inner_mvnn: int,
                 """
mixed_mvnn_parameters = {'num_hidden_layers': 1,
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
                   'init_little_const': 0.1, 
                   'non_mono_num_hidden_layers': 1,
                   'non_mono_num_hidden_units': 20,
                   'non_mono_output_dim': 20, 
                   'non_mono_lin_skip_connection': 0, 
                   'non_mono_dropout_prob': 0,
                   'final_num_hidden_layers': 1,
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
                   'final_lin_skip_connection': 1,
                   'final_output_inner_mvnn': 20,
                   }

#umnn_parameters = {"mon_in": 1, "cond_in": 10, "hiddens": [20,20], "n_out": 1, "nb_steps": 50, "device": "cpu"}
cert_parameters = {"output_parameters": 1, "num_hidden_layers": 4, "hidden_nodes": 20}



def load_dataset(args, num_train_data=100, train_percent=0, seed=100):
    # load dataset using pickle
    # parse filepath
    abspath = "~/masterthesis/ML-CCA-MVNN/compare-models/"
    filepath = "./dataset_generation/datasets/"+ str(args.dataset)+"/"+str(args.dataset)+"_"+str(seed)+"_"+str(args.nbids)+".pkl"
    print("filepath: ", filepath) 
    with open(filepath, "rb") as file:
        print("loaded dataset") 
        dataset = pickle.load(file)
    X = dataset[0]
    y = dataset[1]
    if args.use_dummy:
        X = [bundle+(0,) for bundle in X]

    #in case train percent is not set, use num_train_data
    if train_percent == 0:
        train_percent = num_train_data/len(X)

    #do train val test split
    X_train, test_and_val_X, y_train, test_and_val_y = train_test_split(X, y, train_size=train_percent, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(test_and_val_X,test_and_val_y, test_size=0.9, random_state=1)

    # transform to tensors
    X_train_tensor = torch.FloatTensor(X_train).float()
    y_train_tensor = torch.FloatTensor(y_train).float()
    X_val_tensor = torch.FloatTensor(X_val).float()
    y_val_tensor = torch.FloatTensor(y_val).float()
    X_test_tensor = torch.FloatTensor(X_test).float()
    y_test_tensor = torch.FloatTensor(y_test).float()
    
    print( "X_train_tensor shape is : " , X_train_tensor.size())
    print( "X_val_tensor shape is : " , X_val_tensor.size())
    print( "X_test_tensor shape is : " , X_test_tensor.size())
    if args.scale: 
        max_val = torch.max(y_train_tensor).item() 
        print(max_val, " is max_val " ) 
        y_train_tensor = torch.div(y_train_tensor, max_val)
        y_val_tensor = torch.div(y_val_tensor, max_val)
        y_test_tensor = torch.div(y_test_tensor, max_val)


    #create datasets for dataloader
    return TensorDataset(X_train_tensor, y_train_tensor), TensorDataset(X_val_tensor, y_val_tensor),TensorDataset(X_test_tensor, y_test_tensor)
    """
    else: 
        filepath = "./dataset_generation/experiment_2/blogfeedback/"
        with open(filepath+"blogfeedback_train.pkl","rb") as file: 
            dataset_train = pickle.load(file)

        with open(filepath+"blogfeedback_test.pkl","rb") as file: 
            dataset_test = pickle.load(file)
            """




### UMNN Section ###
#This network needs an embedding Network and a umnn network
#TODO change this from hardcoded
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
    def __init__(self, input_feature, output_features=1,  num_hidden_layers=1, hidden_nodes=20):
        super(MLP_relu, self).__init__()

        self.fc_in = nn.Linear(input_feature, hidden_nodes, bias=True)

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_nodes, hidden_nodes, bias=True) for i in range(num_hidden_layers)])
        self.fc_last = nn.Linear(hidden_nodes, output_features, bias=True)

        self.lam = 10
        self.hidden_nodes = hidden_nodes

    def forward(self, x,y=None,  unscaled=False):
        # scale features to be between 0 and 1
        if unscaled:
            x = F.hardtanh(x, min_val=0.0, max_val=1.0)

        x = self.fc_in(x)
        x = F.relu(x)

        #fun through all the hidden layers
        for i in range(int(len(self.hidden_layers))):
            x = self.hidden_layers[i](x)
            x = F.hardtanh(x, min_val=0.0, max_val=1.0)

        x = self.fc_last(x)
        out = x
        #if self.normalize_regression:
        #    out = F.sigmoid(out)
        return out

    # this is used to decouple betwwen monotone and non monotone
    def reg_forward(self, feature_num, batch_size=512, hidden_nodes=20):
        in_list = []
        out_list = []
        #create a torch random number vector in the intervalk 0 and 1 with size num x feature_num
        input_feature = torch.rand(batch_size, feature_num)

        #input_mono = input_feature[:, :mono_num]
        #input_non_mono = input_feature[:, mono_num:]
        input_feature.requires_grad = True

        # feed through 1st layer and then append input to in_list
        x = self.fc_in(input_feature)
        in_list.append(input_feature)

        x = F.relu(x)
        for i in range(int(len(self.hidden_layers))):
            x = self.hidden_layers[i](x)
            out_list.append(x) # why do we append here already ?

            input_feature = torch.rand(batch_size, hidden_nodes)
            in_list.append(input_feature)
            in_list[-1].requires_grad = True

            #highly unsure whether this makes sense
            x = self.hidden_layers[i](input_feature)
            x = F.relu(x)

        x = self.fc_last(x)
        out_list.append(x)

        return in_list, out_list


class MLP_relu_dummy(nn.Module):
    def __init__(self, mono_feature, non_mono_feature, mono_sub_num=1, non_mono_sub_num=1, mono_hidden_num=5,
                 non_mono_hidden_num=5, compress_non_mono=False, normalize_regression=False):
        super(MLP_relu_dummy, self).__init__()
        self.lam = 10
        self.normalize_regression = normalize_regression
        self.compress_non_mono = compress_non_mono
        if compress_non_mono:
            self.non_mono_feature_extractor = nn.Linear(non_mono_feature, 10, bias=True)
            self.mono_fc_in = nn.Linear(mono_feature + 10, mono_hidden_num, bias=True)
        else:
            self.mono_fc_in = nn.Linear(mono_feature + non_mono_feature, mono_hidden_num, bias=True)

        bottleneck = 10
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

            y = self.non_mono_submods_out[i](y)
            y = F.hardtanh(y, min_val=0.0, max_val=1.0)

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


def get_mvnn(args, input_shape,n_dummy=1):
    capacity_generic_goods = np.array([1 for _ in range(input_shape - 1)])
    model = MVNN_GENERIC(input_dim=input_shape - 1,
                         num_hidden_layers=args.num_hidden_layers,
                         num_hidden_units=args.num_hidden_units,
                         lin_skip_connection=args.lin_skip_connection,
                         dropout_prob=args.dropout_prob,
                         trainable_ts=args.trainable_ts,
                         layer_type=mvnn_parameters['layer_type'],
                         target_max=mvnn_parameters['target_max'],
                         init_method=mvnn_parameters['init_method'],
                         random_ts=mvnn_parameters['random_ts'],
                         init_E=mvnn_parameters['init_E'],
                         init_Var=mvnn_parameters['init_Var'],
                         init_b=mvnn_parameters['init_b'],
                         init_bias=mvnn_parameters['init_bias'],
                         init_little_const=mvnn_parameters['init_little_const'],
                         capacity_generic_goods=capacity_generic_goods,
                         output_size=1,
                         )
    return model
def get_mixed_mvnn(args, input_shape_mono,input_shape_non_mono):
    capacity_generic_goods = np.array([1 for _ in range(input_shape_mono)])
    mono_input = MVNN_GENERIC(input_dim=input_shape_mono,
                         num_hidden_layers=mixed_mvnn_parameters['num_hidden_layers'],
                         num_hidden_units=mixed_mvnn_parameters['num_hidden_units'],
                         layer_type=mixed_mvnn_parameters['layer_type'],
                         target_max=mixed_mvnn_parameters['target_max'],
                         lin_skip_connection=args.lin_skip_connection,
                         dropout_prob=mixed_mvnn_parameters['dropout_prob'],
                         init_method=mixed_mvnn_parameters['init_method'],
                         random_ts=mixed_mvnn_parameters['random_ts'],
                         trainable_ts=mixed_mvnn_parameters['trainable_ts'],
                         init_E=mixed_mvnn_parameters['init_E'],
                         init_Var=mixed_mvnn_parameters['init_Var'],
                         init_b=mixed_mvnn_parameters['init_b'],
                         init_bias=mixed_mvnn_parameters['init_bias'],
                         init_little_const=mixed_mvnn_parameters['init_little_const'],
                         capacity_generic_goods=capacity_generic_goods,
                         final_num_hidden_layers=mixed_mvnn_parameters['num_hidden_layers'],
                         final_num_hidden_units=mixed_mvnn_parameters['num_hidden_units'],
                         final_layer_type=mixed_mvnn_parameters['layer_type'],
                         final_target_max=mixed_mvnn_parameters['target_max'],
                         final_lin_skip_connection=args.final_lin_skip_connection,
                         final_dropout_prob=mixed_mvnn_parameters['dropout_prob'],
                         final_init_method=mixed_mvnn_parameters['init_method'],
                         final_random_ts=mixed_mvnn_parameters['random_ts'],
                         final_trainable_ts=mixed_mvnn_parameters['trainable_ts'],
                         final_init_E=mixed_mvnn_parameters['init_E'],
                         final_init_Var=mixed_mvnn_parameters['init_Var'],
                         final_init_b=mixed_mvnn_parameters['init_b'],
                         final_init_bias=mixed_mvnn_parameters['init_bias'],
                         final_init_little_const=mixed_mvnn_parameters['init_little_const'],
                         final_capacity_generic_goods=capacity_generic_goods
                         )
                         
    return model

umnn_parameters = {"num_embedding_layers": 1, "num_embedding_hiddens": 10, "num_main_hidden_layers" : 1, "num_main_hidden_nodes": 20, "n_out": 1,"nb_steps": 10 }

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


def get_cert(args, train_shape, cert_parameters):
    model = MLP_relu_dummy(mono_feature = train_shape-1,
                            non_mono_feature=1,
                            mono_sub_num = args.num_hidden_layers,
                            non_mono_sub_num = args.num_hidden_layers,
                            mono_hidden_num=args.num_hidden_units,
                            non_mono_hidden_num=args.num_hidden_units,
                            compress_non_mono=args.compress_non_mono,
                            normalize_regression=args.normalize_regression)
    return model


# TODOS:
# choose loss function -> use MSE for now
# store model
# generate plots -> separate File
def train_model(args, model, train, val, test,  metrics,  bidder_id=1, cumm_batch=0, cumm_epoch=0, seed=100, infos=[0,0,0]):
    print("--Starting Training--")
    train_shape = train[0][0].shape[0]
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

            optimizer.zero_grad()
            if args.model == 'CERT':
                n_dummy = 1
                predictions = model.forward(batch[0][:,:-n_dummy], batch[0][:,-n_dummy:])
                cert_loss = loss_mse(predictions.squeeze(1),batch[1][:,bidder_id])
                in_list, out_list = model.reg_forward(train_shape -n_dummy, train_shape - n_dummy)
                reg_loss = generate_regularizer(in_list, out_list)
                loss_tot = cert_loss + model.lam * reg_loss
                loss_tot.backward()
                optimizer.step()

                seed_metrics_train.append([loss_tot.item(),
                           loss_mae(predictions.squeeze(1),batch[1][:,bidder_id]).item(),
                           loss_mse(predictions.squeeze(1),batch[1][:,bidder_id]).item(),
                           loss_evar(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                           loss_medabs(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                           loss_r2(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                           loss_maxerr(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                           loss_mape(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                           loss_d2tw(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                           loss_mpl(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                           loss_d2pl(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                           loss_d2abserr(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                           kendalltau(batch[1][:,bidder_id],predictions.squeeze(1).detach())[0],
                           kendalltau(batch[1][:,bidder_id],predictions.squeeze(1).detach())[1],
                           batch_num,
                           epoch_num, 
                           numpy.array(reg_loss.detach()), 
                           numpy.array(cert_loss.detach()),
                           ])
            elif args.model == 'UMNN':
                model.set_steps(int(torch.randint(1,10, [1])))

                predictions = model.forward(batch[0])
                loss_tot = loss_mse(predictions.squeeze(1),batch[1][:,bidder_id])
                loss_tot.backward()
                optimizer.step()
                seed_metrics_train.append([loss_tot.item(),
                           loss_mae(predictions.squeeze(1),batch[1][:,bidder_id]).item(),
                           loss_mse(predictions.squeeze(1),batch[1][:,bidder_id]).item(),
                           loss_evar(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                           loss_medabs(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                           loss_r2(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                           loss_maxerr(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                           loss_mape(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                           loss_d2tw(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                           loss_mpl(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                           loss_d2pl(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                           loss_d2abserr(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                           kendalltau(batch[1][:,bidder_id],predictions.squeeze(1).detach())[0],
                           kendalltau(batch[1][:,bidder_id],predictions.squeeze(1).detach())[1],
                           batch_num,
                           epoch_num, 
                           reg_loss, 
                           cert_loss,
                           ])

            else:
                predictions = model.forward(batch[0][:,:-1])
                loss_tot = loss_mse(predictions.squeeze(1),batch[1][:,bidder_id])
                loss_tot.backward()
                optimizer.step()
                model.transform_weights()
                seed_metrics_train.append([loss_tot.item(),
                           loss_mae(predictions.squeeze(1),batch[1][:,bidder_id]).item(),
                           loss_mse(predictions.squeeze(1),batch[1][:,bidder_id]).item(),
                           loss_evar(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                           loss_medabs(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                           loss_r2(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                           loss_maxerr(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                           loss_mape(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                           loss_d2tw(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                           loss_mpl(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                           loss_d2pl(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                           loss_d2abserr(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                           kendalltau(batch[1][:,bidder_id],predictions.squeeze(1).detach())[0],
                           kendalltau(batch[1][:,bidder_id],predictions.squeeze(1).detach())[1],
                           batch_num,
                           epoch_num, 
                           reg_loss, 
                           cert_loss,
                           ])


        ### Validation ###
        #print("START validation")
        val_size = len(val)
        print(" val_size is :" , val_size)
        val_loader = torch.utils.data.DataLoader(val, batch_size=val_size, shuffle=True)
        for batch in tqdm(val_loader):
            #batch = batch.to(device)
            if args.model == "MVNN": 
                predictions = model.forward(batch[0][:,:-1])

            elif args.model == "UMNN":
                predictions = model.forward(batch[0])

            else :
                predictions = model.forward(batch[0][:, :-n_dummy], batch[0][:, -n_dummy:])
            val_loss_tot = loss_mse(predictions.squeeze(1), batch[1][:, bidder_id])
            #print("Val loss is : " ,val_loss.item())

            seed_metrics_val.append([val_loss_tot.item(),
                       loss_mae(predictions.squeeze(1),batch[1][:,bidder_id]).item(),
                       loss_mse(predictions.squeeze(1),batch[1][:,bidder_id]).item(),
                       loss_evar(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       loss_medabs(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       loss_r2(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       loss_maxerr(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       loss_mape(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       loss_d2tw(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       loss_mpl(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       loss_d2pl(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       loss_d2abserr(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       kendalltau(batch[1][:,bidder_id],predictions.squeeze(1).detach())[0],
                       kendalltau(batch[1][:,bidder_id],predictions.squeeze(1).detach())[1],
                       batch_num,
                       epoch_num, 
                       ])
            """
            wandb.log({"val_loss": val_loss.item(),
                       "val_loss_mean_absolute_error": loss_mae(predictions.squeeze(1),batch[1][:,bidder_id]).item(),
                       "val_loss_mse": loss_mse(predictions.squeeze(1),batch[1][:,bidder_id]).item(),
                       "val_loss_explained_variance_score": loss_evar(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       "val_loss_median_absolute_err": loss_medabs(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       "val_loss_r2": loss_r2(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       "val_loss_max_err": loss_maxerr(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       "val_loss_mean absolute_percentage_err": loss_mape(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       "val_loss_d2_tweedie_score": loss_d2tw(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       "val_loss_mean_pinball_loss": loss_mpl(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       "val_loss_d2_pinball_score": loss_d2pl(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       "val_loss_d2_absolute_err_score": loss_d2abserr(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       "val_kendall_tau_statistics": kendalltau(batch[1][:,bidder_id],predictions.squeeze(1).detach())[0],
                       "val_kendall_tau_p_val": kendalltau(batch[1][:,bidder_id],predictions.squeeze(1).detach())[1],
                       "Batch_num":  batch_num,
                       "Epoch":  epoch_num})
            """
    print("END train and validation")
    print("Start Testing")
        ### Test ###
    test_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=True)
    for batch in tqdm(test_loader):
        #batch = batch.to(device)
        if args.model == "MVNN":
            predictions = model.forward(batch[0][:,:-1])
        elif args.model == "UMNN":
            predictions = model.forward(batch[0])
        else:
            predictions = model.forward(batch[0][:, :-n_dummy], batch[0][:, -n_dummy:])
        test_loss_tot = loss_mse(predictions.squeeze(1), batch[1][:, bidder_id])
        #print("test loss is : ", test_loss.item())
        seed_metrics_test.append([test_loss_tot.item(),
                       loss_mae(predictions.squeeze(1),batch[1][:,bidder_id]).item(),
                       loss_mse(predictions.squeeze(1),batch[1][:,bidder_id]).item(),
                       loss_evar(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       loss_medabs(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       loss_r2(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       loss_maxerr(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       loss_mape(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       loss_d2tw(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       loss_mpl(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       loss_d2pl(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       loss_d2abserr(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       kendalltau(batch[1][:,bidder_id],predictions.squeeze(1).detach())[0],
                       kendalltau(batch[1][:,bidder_id],predictions.squeeze(1).detach())[1],
                       batch_num,
                       epoch_num, 
                       ])
        """
        wandb.log({"test_loss": test_loss.item(),
                   "test_loss_mean_absolute_error": loss_mae(predictions.squeeze(1), batch[1][:, bidder_id]).item(),
                   "test_loss_mse": loss_mse(predictions.squeeze(1), batch[1][:, bidder_id]).item(),
                   "test_loss_explained_variance_score": loss_evar(y_true=batch[1][:, bidder_id],
                                                                   y_pred=predictions.squeeze(1).detach()).item(),
                   "test_loss_median_absolute_err": loss_medabs(y_true=batch[1][:, bidder_id],
                                                                y_pred=predictions.squeeze(1).detach()).item(),
                   "test_loss_r2": loss_r2(y_true=batch[1][:, bidder_id], y_pred=predictions.squeeze(1).detach()).item(),
                   "test_loss_max_err": loss_maxerr(y_true=batch[1][:, bidder_id],
                                                    y_pred=predictions.squeeze(1).detach()).item(),
                   "test_loss_mean absolute_percentage_err": loss_mape(y_true=batch[1][:, bidder_id],
                                                                       y_pred=predictions.squeeze(1).detach()).item(),
                   "test_loss_d2_tweedie_score": loss_d2tw(y_true=batch[1][:, bidder_id],
                                                           y_pred=predictions.squeeze(1).detach()).item(),
                   "test_loss_mean_pinball_loss": loss_mpl(y_true=batch[1][:, bidder_id],
                                                           y_pred=predictions.squeeze(1).detach()).item(),
                   "test_loss_d2_pinball_score": loss_d2pl(y_true=batch[1][:, bidder_id],
                                                           y_pred=predictions.squeeze(1).detach()).item(),
                   "test_loss_d2_absolute_err_score": loss_d2abserr(y_true=batch[1][:, bidder_id],
                                                                    y_pred=predictions.squeeze(1).detach()).item(),
                   "test_kendall_tau_statistics": kendalltau(batch[1][:, bidder_id], predictions.squeeze(1).detach())[0],
                   "test_kendall_tau_p_val": kendalltau(batch[1][:, bidder_id], predictions.squeeze(1).detach())[1]})
    print("End Testing")
    """

    metrics[0].append(seed_metrics_train)
    metrics[1].append(seed_metrics_val)
    metrics[2].append(seed_metrics_test)
    infos[0] = batch_num
    infos[1] = epoch_num
    return model, metrics, infos

def get_model(args, train_shape):
    ### define model ###
    if args.model == 'MVNN':
        model = get_mvnn(args, train_shape)

    elif args.model == 'UMNN':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Device is : ", device)
        model = get_umnn(umnn_parameters, train_shape, device)

    elif args.model == 'CERT':
        print("LOADING CERT")
        model = get_cert(args=args, train_shape=train_shape, cert_parameters=cert_parameters)
    return  model

def log_metrics(args, metrics):
    """
    wandb.define_metric("Batch_num")
    wandb.define_metric("Epoch")
    wandb.define_metric("loss_tot", step_metric="Batch_num")
    wandb.define_metric("loss_mse", step_metric="Batch_num")
    wandb.define_metric("loss_mae", step_metric="Batch_num")
    wandb.define_metric("loss_evar", step_metric="Batch_num")
    wandb.define_metric("loss_medabs", step_metric="Batch_num")
    wandb.define_metric("loss_r2", step_metric="Batch_num")
    wandb.define_metric("loss_maxerr", step_metric="Batch_num")
    wandb.define_metric("loss_mape", step_metric="Batch_num")
    wandb.define_metric("loss_d2tw", step_metric="Batch_num")
    wandb.define_metric("loss_mpl", step_metric="Batch_num")
    wandb.define_metric("loss_d2pl", step_metric="Batch_num")
    wandb.define_metric("loss_d2abserr", step_metric="Batch_num")
    wandb.define_metric("kendall_tau_statistics", step_metric="Batch_num")
    wandb.define_metric("kendall_tau_p_val", step_metric="Batch_num")
    wandb.define_metric("val_loss_tot", step_metric="Batch_num")
    wandb.define_metric("val_loss_mse", step_metric="Batch_num")
    wandb.define_metric("val_loss_mae", step_metric="Batch_num")
    wandb.define_metric("val_loss_evar", step_metric="Batch_num")
    wandb.define_metric("val_loss_medabs", step_metric="Batch_num")
    wandb.define_metric("val_loss_r2", step_metric="Batch_num")
    wandb.define_metric("val_loss_maxerr", step_metric="Batch_num")
    wandb.define_metric("val_loss_mape", step_metric="Batch_num")
    wandb.define_metric("val_loss_d2tw", step_metric="Batch_num")
    wandb.define_metric("val_loss_mpl", step_metric="Batch_num")
    wandb.define_metric("val_loss_d2pl", step_metric="Batch_num")
    #wandb.define_metric("val_loss_d2abserr", step_metric="Batch_num")
    #mets_array = np.array(metrics)
    """

    
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
                   "Cert_regularizer": mean_train[i,16],
                   "Cert_loss": mean_train[i,17],
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

    # log  parameters 
    group_id = str(args.model) + str(args.dataset) + str(args.bidder_id)
    run_id = group_id  + "Rand_id_"+ str(np.random.randint(300))

    run.name = run_id
    #run.group = group_id

    if args.model == "MVNN": 
        wandb.config.update(mvnn_parameters, allow_val_change=True)

    elif args.model == "UMNN":    
        wandb.config.update(umnn_parameters, allow_val_change=True)

    else:        
        wandb.config.update(cert_parameters, allow_val_change=True)
    

    #hard set to 0 non monotone variables 
    #if args.model == "MVNN":
    #    args.use_dummy = False


    # define metrics 
    metrics = [[],[],[]]
    infos = [0,0,0] # cumulative batch and epoch last is bool if CERT is extended  


    # TODO remove bidder id loop 
    #bidder_ids = [args.bidder_id]
    #for bidder in bidder_ids:
    for num, seed in enumerate(range(args.initial_seed, args.initial_seed + args.num_seeds)):
        wandb.log({"model": args.model, "dataset": args.dataset})


        ### load dataset ###
        train, val, test = load_dataset(args, train_percent=args.train_percent,seed=seed)
        train_shape = train[0][0].shape[0]

        print(train_shape, " is the train shape and seed is ", seed )
        #print("--- Loaded dataset successfully ---")

        ### define model ###
        model = get_model(args, train_shape)
        #print("--- Loaded model successfully ---")


        ### train model ###
        model, metrics, infos = train_model(args, model, train, val,test,  metrics, bidder_id=args.bidder_id, seed=seed, infos=infos)
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
    ### log metrics ###
    #log_metrics(args, metrics)

    

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
    #MODEL = "MVNN"
    MODEL = "CERT"
    #MODEL = "UMNN"
    print("Running model: ", MODEL)

    #wandb.init(project="MVNN-Runs")
    #wandb.init(project="MVNN-Runs", config={"n_runs": 0 }, reinit=True)
    #wandb.config.update(args, allow_val_change=True)

    sweep_config = { 
        "method": "random", 
        "metric": {"goal": "minimize", "name": "val_loss_tot"}, 
        "parameters": {
            "learning_rate": {"values": [0.001, 0.005, 0.01,  0.05, 0.1]},
            "num_hidden_layers": { "values" : [1,2,3]},
            "num_hidden_units": { "values": [16,32,64,128,256]},
            "batch_size": { "values": [25,50,100]},
            "l2_rate": { "values": [0.0, 0.5, 1.0]},
            "model": {"values":[str(MODEL)]},
            "dataset": {"values":["lsvm"]}, 
            "bidder_id":{ "values": [0]},
            "epochs":{ "values": [100]},
            #"dataset": {"values":["gsvm", "lsvm","srvm","mrvm"]}, 
            # MVNN Params
            #"lin_skip_connection": {"values": ["True", "False"]},
            #"trainable_ts": {"values": ["True", "False"]},
            #"dropout_prob": {"values": [0., 0.1, 0.2, 0.3, 0.4 ,0.5]},
            #CERT Params
            "compress_non_mono": {"values": ["True", "False"]},
            "normalize_regression": {"values": ["True", "False"]},
            },
        }

    sweep_id = wandb.sweep(sweep=sweep_config, project="Mono HPO lsvm parallel")
    #wandb.agent(sweep_id, function=main, count=35)
    with mp.Pool(24) as p : 
        p.map(wandb.agent(sweep_id, function=main, count=10))


    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #print("Device is : " , device)

    #print("Testing classic Main") 
    #parser = init_parser()
    #args = parser.parse_args()
    #main(args)
    exit()

