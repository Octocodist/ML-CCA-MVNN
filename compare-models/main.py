import os




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
from config import cert_parameters, mvnn_parameters, umnn_parameters



def init_parser():
    parser = argparse.ArgumentParser()

    ### experiment parameters ###
    parser.add_argument("--dataset", help="dataset to use", default="mrvm", choices=['gsvm' , 'lsvm', 'srvm', 'mrvm'] )
    parser.add_argument("--nbids", help="number of bids to use", default=25000)
    parser.add_argument("--bidder_id", help="bidder id to use", default=0)
    parser.add_argument('-m','--model',  type=str, help='Choose model to train: UMNN, MVNN', choices=['UMNN','MVNN','CERT'], default='CERT')
    parser.add_argument("-tp","--train_percent", type=float, default=0.2, help="percentage of data to use for training")
    parser.add_argument("-ud","--use_dummy", type=bool, default=True, help="use dummy dataset")
    parser.add_argument("-ns","--num_seeds", type=int, default=10, help="number of seeds to use for hpo")
    parser.add_argument("-is","--initial_seed", type=int, default=100, help="initial seed to use for hpo")

    ### training parameters ###
    parser.add_argument("--epochs", help="number of epochs to train", default=100)
    parser.add_argument("--batch_size", help="batch size to use", default=32)
    parser.add_argument("--learning_rate", help="learning rate", default=0.001)
    #parser.add_argument("--loss", help="ltenary operator expression c++oss function to use", default="mse")
    #parser.add_argument("--optimizer", help="optimizer to use", default="adam")

    ### model parameters ###
    parser.add_argument("--num_hidden_layers", help="number of hidden layers", default=1)
    parser.add_argument("--num_hidden_units", help="number of hidden units", default=20)
    parser.add_argument("--layer_type", help="layer type", default="MVNNLayerReLUProjected")
    parser.add_argument("--target_max", help="target max", default=1)
    parser.add_argument("--lin_skip_connection", type=bool,  help="linear skip connection", default=False)
    parser.add_argument("--dropout_prob", help="dropout probability", default=0)
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

umnn_parameters = {"mon_in": 10, "cond_in": 0, "hiddens": [20,20], "n_out": 1, "nb_steps": 50, "device": "cpu"}

#cert_parameters = {"output_parameters": 1, "num_hidden_layers": 4, "hidden_nodes": 20}



def load_dataset(args, num_train_data=1000, train_percent=0, seed=100):
    # load dataset using pickle
    # parse filepath

    filepath = "./dataset_generation/datasets/"+ str(args.dataset)+"/"+str(args.dataset)+"_"+str(seed)+"_"+str(args.nbids)+".pkl"
    with open(filepath, "rb") as file:
        dataset = pickle.load(file)
    X = dataset[0]
    y = dataset[1]
    if args.use_dummy:
        X = [bundle+(0,) for bundle in X]

    #in case train percent is not set, use num_train_data
    if train_percent == 0:
        train_percent = len(X)/num_train_data

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

    #create datasets for dataloader
    return TensorDataset(X_train_tensor, y_train_tensor), TensorDataset(X_val_tensor, y_val_tensor),TensorDataset(X_test_tensor, y_test_tensor)




### UMNN Section ###
#This network needs an embedding Network and a umnn network
#TODO change this from hardcoded
class EmbeddingNet(nn.Module):
    def __init__(self, in_embedding, in_main, out_embedding, device='cpu',num_embedding_layers=3, num_hidden_nodes=200):
        super(EmbeddingNet, self).__init__()
        self.embedding_net = nn.Sequential(nn.Linear(in_embedding, 200), nn.ReLU(),
                                           nn.Linear(200, 200), nn.ReLU(),
                                           nn.Linear(200, out_embedding), nn.ReLU()).to(device)

        self.umnn = SlowDMonotonicNN(in_main, cond_in=out_embedding, hiddens=[100, 100, 100], n_out=1, nb_steps= 300, device= device)

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

    def forward(self, x,y=None,  unscaled=True):
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
                         capacity_generic_goods=capacity_generic_goods
                         )
    return model

def get_umnn(umnn_parameters, input_shape, n_dummy=1, n_items=18):
    model = EmbeddingNet(in_embedding =input_shape, in_main=input_shape, out_embedding=n_dummy, device="cpu")
    return model
def get_cert(args, train_shape, cert_parameters):
    if args.use_dummy:
        model = MLP_relu_dummy(train_shape-1, 1,1,1,5,5,False, False)
    else:
        model = MLP_relu(train_shape, cert_parameters["output_parameters"], cert_parameters["num_hidden_layers"], cert_parameters["hidden_nodes"])
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
def train_model(model, train, train_shape, val, test, bidder_id=1, n_dummy=1, batch_size=64):
    # metrics for regression
    loss_mse = torch.nn.MSELoss()
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
    wandb.define_metric("Batch_num")
    wandb.define_metric("Epoch")

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)




    epochs = 20

    optimizer = Adam(model.parameters())

    wandb.watch(model, log="all")
    wandb.config.update(args, allow_val_change=True)
    wandb.config.update({'optimizer': optimizer})


    batch_num = 0

    for e in tqdm(range(epochs)):
        for batch in train_loader:
            batch_num +=1

            optimizer.zero_grad()
            if args.model == 'CERT':
                predictions = model.forward(batch[0][:,:-n_dummy], batch[0][:,-n_dummy:])
                loss = loss_mse(predictions.squeeze(1),batch[1][:,bidder_id])
                in_list, out_list = model.reg_forward(train_shape, train_shape - n_dummy)
                reg_loss = generate_regularizer(in_list, out_list)
                loss_tot = loss + model.lam * reg_loss
                loss_tot.backward()
                optimizer.step()


            elif args.model == 'UMNN':
                model.set_steps(int(torch.randint(30,60, [1])))
                predictions = model.forward(batch[0])
                loss = loss_mse(predictions.squeeze(1),batch[1][:,bidder_id])

                loss.backward()
                optimizer.step()
            else:
                predictions = model.forward(batch[0])
                loss = loss_mse(predictions.squeeze(1),batch[1][:,bidder_id])

                loss.backward()
                optimizer.step()
                model.transform_weights()

            wandb.log({"loss": loss.item(),
                       "loss_mean_absolute_error": loss_mae(predictions.squeeze(1),batch[1][:,bidder_id]).item(),
                       "loss_mse": loss_mse(predictions.squeeze(1),batch[1][:,bidder_id]).item(),
                       "loss_explained_variance_score": loss_evar(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       "loss_median_absolute_err": loss_medabs(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       "loss_r2": loss_r2(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       "loss_max_err": loss_maxerr(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       "loss_mean absolute_percentage_err": loss_mape(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       "loss_d2_tweedie_score": loss_d2tw(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       "loss_mean_pinball_loss": loss_mpl(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       "loss_d2_pinball_score": loss_d2pl(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       "loss_d2_absolute_err_score": loss_d2abserr(y_true=batch[1][:,bidder_id],y_pred=predictions.squeeze(1).detach()).item(),
                       "kendall_tau_statistics": kendalltau(batch[1][:,bidder_id],predictions.squeeze(1).detach())[0],
                       "kendall_tau_p_val": kendalltau(batch[1][:,bidder_id],predictions.squeeze(1).detach())[1],
                       "Batch_num":  batch_num,
                       "Epoch":  e})

        ### Validation ###
        print("START validation")
        val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True)
        for batch in val_loader:
            if args.model == "MVNN":
                predictions = model.forward(batch[0])
            else :
                predictions = model.forward(batch[0][:, :-n_dummy], batch[0][:, -n_dummy:])
            val_loss = loss_mse(predictions.squeeze(1), batch[1][:, bidder_id])
            print("Val loss is : " ,val_loss.item())
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
                       "Epoch":  e})

        print("END validation")

    print("Start Testing")
        ### Test ###
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)
    for batch in test_loader:
        if args.model == "MVNN":
            predictions = model.forward(batch[0])
        else:
            predictions = model.forward(batch[0][:, :-n_dummy], batch[0][:, -n_dummy:])
        test_loss = loss_mse(predictions.squeeze(1), batch[1][:, bidder_id])
        print("test loss is : ", test_loss.item())

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

    return model

def main(args):
    print("--Start Program--")
    print("We are training over:", args.num_seeds, " seeds")

    wandb.init(project="MVNN-Runs", config={"n_runs": 0 }, reinit=True)
    wandb.config.update(args, allow_val_change=True)


    for num, seed in enumerate(range(args.initial_seed, args.initial_seed + args.num_seeds)):
        #set run_id 
        group_id = str(args.model) + str(args.dataset) + str(args.bidder_id)
        run_id = group_id + str(seed) 
        wandb.init(project="MVNN-Runs",id=run_id, group = group_id , config={"n_run": num}, reinit=True)
        wandb.config.update(cert_parameters, allow_val_change=True)



        ### load dataset ###
        train, val, test = load_dataset(args, train_percent=args.train_percent,seed=seed)
        train_shape = train[0][0].shape[0]

        print(train_shape, " is the train shape and seed is ", seed)
        print("--- Loaded dataset successfully ---")


        ### define model ###
        if args.model == 'MVNN':
            model = get_mvnn(args,train_shape)
            print("MVNN loaded")
            wandb.config.update(mvnn_parameters, allow_val_change=True)
            model = train_model(model, train, train_shape, val, test)
        elif args.model == 'UMNN':
            model = get_umnn(umnn_parameters,train_shape)
            print("UMNN loaded")
            wandb.config.update(umnn_parameters, allow_val_change=True)
            model = train_model(model, train, train_shape, val, test)
        elif args.model == 'CERT':
            model = get_cert(args, train_shape, cert_parameters)
            print("CERT loaded")
            mono_flag = False
            while not mono_flag:
                model = train_model(model, train, train_shape, val, test)
                # certify first layer
                print("Certifying network!")
                assert(args.use_dummy)
                n_dummy = 1
                print("Start Certification") 
                mono_flag = certify_neural_network(model, train_shape-n_dummy)
                if not mono_flag:
                    model.lam *= 10
                    print("Network not monotonic, increasing regularization strength to ", model.lam)
                    wandb.log({"lam":model.lam})

                    if model.lam == 1000000000:
                        print("Exiting because of too many trys in CERT") 
                        mono_flag = True
        else:
            print("Model not implemented yet")
            exit(1234)

if __name__ == "__main__":
    print("--Start Parsing Arguments--")
    parser = init_parser()
    args = parser.parse_args()


    if args.model == "MVNN":
        args.use_dummy = False

    #os.environ['WANDB_SILENT'] = "true"
    os.environ['WANDB_MODE'] = "offline"
    #wandb.init(project="MVNN-Runs")
    #wandb.init(project="MVNN-Runs", config={"n_runs": 0 }, reinit=True)
    #wandb.config.update(args, allow_val_change=True)

    main(args)
    wandb.finish()

