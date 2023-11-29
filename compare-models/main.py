import os

import numpy
import pickle
import torch.nn as nn

import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from torch.optim import Adam
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset
import torch.nn.functional as F



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



def init_parser():
    parser = argparse.ArgumentParser()

    ### experiment parameters ###
    parser.add_argument("--dataset", help="dataset to use", default="lsvm")
    parser.add_argument("--nbids", help="number of bids to use", default=20)
    parser.add_argument("--bidder_id", help="bidder id to use", default=3)
    parser.add_argument('-m','--model',  type=str, help='Choose model to train: UMNN, MVNN', choices=['UMNN','MVNN','CERT'], default='UMNN')
    parser.add_argument("-tp","--train_percent", type=float, default=0.1, help="percentage of data to use for training")
    parser.add_argument("-ud","--use_dummy", type=bool, default=True, help="use dummy dataset")

    ### training parameters ###
    parser.add_argument("--epochs", help="number of epochs to train", default=100)
    parser.add_argument("--batch_size", help="batch size to use", default=32)
    parser.add_argument("--lr", help="learning rate", default=0.001)
    #parser.add_argument("--loss", help="ltenary operator expression c++oss function to use", default="mse")
    #parser.add_argument("--optimizer", help="optimizer to use", default="adam")

    ### model parameters ###
    parser.add_argument("--num_hidden_layers", help="number of hidden layers", default=1)
    parser.add_argument("--num_hidden_units", help="number of hidden units", default=20)
    parser.add_argument("--layer_type", help="layer type", default="MVNNLayerReLUProjected")
    parser.add_argument("--target_max", help="target max", default=1)
    parser.add_argument("--lin_skip_connection", help="linear skip connection", default=1)
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
MVNN_parameters = {'num_hidden_layers': 1,
                    'num_hidden_units': 20,
                    'layer_type': 'MVNNLayerReLUProjected',
                    'target_max': 1, # TODO: check
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
umnn_parameters = {"mon_in": 10, "cond_in": 0, "hiddens": [10,10], "n_out": 1, "nb_steps": 50, "device": "cpu"}

CERT_parameters = {"output_parameters": 1, "num_hidden_layers": 4, "hidden_nodes": 20}



def load_dataset(args, num_train_data=1000, train_percent=0):
    # load dataset using pickle
    # parse filepath
    filepath = "./dataset_generation/datasets/"+ str(args.dataset)+"/"+str(args.dataset)+"_"+str(args.bidder_id)+"_"+str(args.nbids)+".pkl"
    with open(filepath, "rb") as file:
        dataset = pickle.load(file)
    if args.dataset == "gsvm":
        N = 7
        M = 18
    if args.dataset == "lsvm":
        N = 6
        M = 18
    if args.dataset == "srvm":
        N = 7
        M = 29
    if args.dataset == "mrvm":
        N = 10
        M = 98
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
    def __init__(self, in_embedding, in_main, out_embedding, device='cpu'):
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


def get_mvnn(input_shape, n_dummy=1):
    capacity_generic_goods = np.array([1 for _ in range(input_shape-n_dummy)])
    model = MVNN_GENERIC(input_dim=input_shape - n_dummy,
                         num_hidden_layers=MVNN_parameters['num_hidden_layers'],
                         num_hidden_units=MVNN_parameters['num_hidden_units'],
                         layer_type=MVNN_parameters['layer_type'],
                         target_max=MVNN_parameters['target_max'],
                         lin_skip_connection=MVNN_parameters['lin_skip_connection'],
                         dropout_prob=MVNN_parameters['dropout_prob'],
                         init_method=MVNN_parameters['init_method'],
                         random_ts=MVNN_parameters['random_ts'],
                         trainable_ts=MVNN_parameters['trainable_ts'],
                         init_E=MVNN_parameters['init_E'],
                         init_Var=MVNN_parameters['init_Var'],
                         init_b=MVNN_parameters['init_b'],
                         init_bias=MVNN_parameters['init_bias'],
                         init_little_const=MVNN_parameters['init_little_const'],
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
def train_model(model, train, train_shape, val, test):
    batch_size = 32*2
    bidder_id = 1
    n_dummy = 1

    # metrics for regression
    loss_mse = torch.nn.MSELoss()
    loss_mae = torch.nn.L1Loss()

    # metrics for classification
    #loss_cert = torch.nn.BCEWithLogitsLoss()
    #loss_NLL = torch.nn.NLLLoss()
    #loss_CrossEntropy = torch.nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    epochs = 13

    optimizer = Adam(model.parameters())

    wandb.watch(model, log="all")
    wandb.config.update(args)
    wandb.config.update(MVNN_parameters)
    wandb.config.update({'optimizer': optimizer})
    wandb.config.update({'loss': loss_mse})
    wandb.config.update({'train_percent': args.train_percent})

    for e in range(epochs):
        for batch in tqdm(train_loader):
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
                loss = loss_mse(predictions.squeeze(1),batch[1][:,1])

                loss.backward()
                optimizer.step()
            else:
                predictions = model.forward(batch[0])
                loss = loss_mse(predictions,batch[1][:,1])

                loss.backward()
                optimizer.step()
                model.transform_weights()

            wandb.log({"loss": loss.item()})
            wandb.log({"loss_mae":loss_mae(predictions.squeeze(1),batch[1][:,1]).item()})



        ### Validation ###
        print("START validation")
        val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True)
        for batch in val_loader:
            #predictions = model.forward(batch[0])
            predictions = model.forward(batch[0][:, :-n_dummy], batch[0][:, -n_dummy:])
            val_loss = loss_mse(predictions.squeeze(1), batch[1][:, 1])
            wandb.log({"val_loss": loss.item()})

        ### Test ###
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)
        for batch in test_loader:
            predictions = model.forward(batch[0][:, :-n_dummy], batch[0][:, -n_dummy:])
            #predictions = model.forward(batch[0])
            val_loss = loss_mse(predictions.squeeze(1), batch[1][:, 1])
            wandb.log({"test_loss": loss.item()})
        print("END validation")
    return model

def main(args):
    print("--Start Program--")

    ### load dataset ###
    train, val, test = load_dataset(args, train_percent=args.train_percent)
    train_shape = train[0][0].shape[0]

    print(train_shape, " is the train shape")
    print("--- Loaded dataset successfully ---")


    ### define model ###
    if args.model == 'MVNN':
        model = get_mvnn(train_shape)
        print("MVNN loaded")
        model = train_model(model, train, train_shape, val, test)
    elif args.model == 'UMNN':
        model = get_umnn(umnn_parameters,train_shape)
        print("UMNN loaded")
        model = train_model(model, train, train_shape, val, test)
    elif args.model == 'CERT':
        model = get_cert(args, train_shape, CERT_parameters)
        print("CERT loaded")
        mono_flag = False
        while not mono_flag:
            model = train_model(model, train, train_shape, val, test)
            # certify first layer
            print("Certifying network!")
            mono_flag = certify_grad_with_gurobi(model.fc_in, model.hidden_layers[0], train_shape)
            for i in range(1, CERT_parameters["num_hidden_layers"]-2, 2):
                curr_flag = certify_grad_with_gurobi(model.hidden_layers[i], model.hidden_layers[i + 1],
                                                     train_shape)
                if mono_flag and curr_flag:
                    mono_flag = True
                else:
                    mono_flag = False
            final_flag = certify_grad_with_gurobi(model.hidden_layers[-1], model.
                                                  fc_last, train_shape)
            if mono_flag and final_flag:
                mono_flag = True
            else:
                mono_flag = False
                model.lam *= 10
                print("Network not monotonic, increasing regularization strength to ", model.lam)
    else:
        print("Model not implemented yet")
        exit(1234)

if __name__ == "__main__":
    print("--Start Parsing Arguments--")
    parser = init_parser()
    args = parser.parse_args()

    args.bidder_id = int(1)
    args.dataset = "gsvm"
    args.nbids = int(25000)

    #os.environ['WANDB_SILENT'] = "true"
    #os.environ['WANDB_MODE'] = "offline"
    wandb.init(project="mvnn")
    wandb.config.update(args)

    main(args)

