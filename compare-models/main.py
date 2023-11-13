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


import wandb

### custom imports ###
from mvnns.mvnn import MVNN
from mvnns.mvnn_generic import MVNN_GENERIC
from generalized_UMNN.models.MultidimensionnalMonotonicNN import SlowDMonotonicNN



def init_parser():
    parser = argparse.ArgumentParser()

    ### experiment parameters ###
    parser.add_argument("--dataset", help="dataset to use", default="lsvm")
    parser.add_argument("--nbids", help="number of bids to use", default=20)
    parser.add_argument("--bidder_id", help="bidder id to use", default=3)
    parser.add_argument('-m','--model',  type=str, help='Choose model to train: UMNN, MVNN', choices=['UMNN','MVNN'], default='UMNN')
    parser.add_argument("-tp","--train_percent", type=float, default=0.1, help="percentage of data to use for training")

    ### training parameters ###
    parser.add_argument("--epochs", help="number of epochs to train", default=100)
    parser.add_argument("--batch_size", help="batch size to use", default=32)
    parser.add_argument("--lr", help="learning rate", default=0.001)
    #parser.add_argument("--loss", help="loss function to use", default="mse")
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



def load_dataset(args, num_train_data=1000, train_percent=0):
    # load dataset using pickle
    # parse filepath
    filepath = "./dataset_generation/datasets/"+ str(args.dataset)+"/"+str(args.dataset)+"_"+str(args.bidder_id)+"_"+str(args.nbids)+".pkl"
    with open(filepath, "rb") as file:
        dataset = pickle.load(file)
    #split dataset into train and test
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

def get_mvnn(input_shape):
    num_hidden_layers = MVNN_parameters['num_hidden_layers']
    num_hidden_units = MVNN_parameters['num_hidden_units']
    layer_type = MVNN_parameters['layer_type']
    target_max = MVNN_parameters['target_max']  # TODO: check
    lin_skip_connection = MVNN_parameters['lin_skip_connection']
    dropout_prob = MVNN_parameters['dropout_prob']
    init_method = MVNN_parameters['init_method']
    random_ts = MVNN_parameters['random_ts']
    trainable_ts = MVNN_parameters['trainable_ts']
    init_E = MVNN_parameters['init_E']
    init_Var = MVNN_parameters['init_Var']
    init_b = MVNN_parameters['init_b']
    init_bias = MVNN_parameters['init_bias']
    init_little_const = MVNN_parameters['init_little_const']
    #capacity_generic_goods = MVNN_parameters['capacity_generic_goods']
    #hard coded for gsvm
    capacity_generic_goods = np.array([1 for _ in range(18)])
    model = MVNN_GENERIC(input_dim=input_shape,
                    num_hidden_layers = num_hidden_layers,
                    num_hidden_units = num_hidden_units,
                    layer_type = layer_type,
                    target_max = target_max,
                    lin_skip_connection = lin_skip_connection,
                    dropout_prob = dropout_prob,
                    init_method = init_method,
                    random_ts = random_ts,
                    trainable_ts = trainable_ts,
                    init_E = init_E,
                    init_Var = init_Var,
                    init_b = init_b,
                    init_bias = init_bias,
                    init_little_const = init_little_const,
                    capacity_generic_goods=capacity_generic_goods
                    )
    return model

#This network needs an embedding Network and a umnn network
#TODO change this from hardcoded
class EmbeddingNet(nn.Module):
    def __init__(self, in_embedding, in_main, out_embedding, device):
        super(EmbeddingNet, self).__init__()
        self.embedding_net = nn.Sequential(nn.Linear(in_embedding, 200), nn.ReLU(),
                                           nn.Linear(200, 200), nn.ReLU(),
                                           nn.Linear(200, out_embedding), nn.ReLU()).to(device)

        self.umnn = SlowDMonotonicNN(in_main, out_embedding, [100, 100, 100], 1, 300, device)

    def set_steps(self, nb_steps):
        self.umnn.set_steps(nb_steps)

    def forward(self, x):
        h = self.embedding_net(x[:,:])
        #h = self.embedding_net(x)
        return torch.sigmoid(self.umnn(x, h))
def get_umnn(umnn_parameters, input_shape):
    model = EmbeddingNet(in_embedding =input_shape, in_main=input_shape, out_embedding=18, device="cpu")

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

def main(args):
    print("--Start Program--")

    ### load dataset ###
    train, val, test = load_dataset(args, train_percent=args.train_percent)
    train_shape = train[0][0].shape[0]
    print("--- Loaded dataset successfully ---")

    ### define model ###
    if args.model == 'MVNN':
        model = get_mvnn(train_shape)
    elif args.model == 'UMNN':
        model = get_umnn(umnn_parameters,train_shape)
        print("UMNN loaded")
    else:
        print("Model not implemented yet")
        exit(1234)

    ### define loss function ###
    loss_mse = torch.nn.MSELoss()

    batch_size = 12

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    epochs = 1
    optimizer = Adam(model.parameters())

    ### wandb ###
    wandb.watch(model, log="all")
    wandb.config.update(args)
    wandb.config.update(MVNN_parameters)
    wandb.config.update({'optimizer': optimizer})
    wandb.config.update({'loss': loss_mse})
    wandb.config.update({'train_percent': args.train_percent})

    ### Training ###
    for e in tqdm(range(epochs)):
        for batch in train_loader:
            optimizer.zero_grad()

            predictions = model.forward(batch[0])

            loss = loss_mse(predictions,batch[1][:,1])

            loss.backward()
            optimizer.step()
            if args.model == 'MVNN':
                model.transform_weights()
            elif args.model == 'UMNN':
                model.set_steps(int(torch.randint(30,60, [1])))


            wandb.log({"loss": loss.item()})

    ### Validation ###
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True)
    for batch in val_loader:
        predictions = model.forward(batch[0])
        loss = loss_mse(predictions,batch[1][:,1])
        wandb.log({"val_loss": loss.item()})


    ### Test ###
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)
    for batch in test_loader:
        predictions = model.forward(batch[0])
        loss = loss_mse(predictions,batch[1][:,1])
        wandb.log({"test_loss": loss.item()})


if __name__ == "__main__":
    print("--Start Parsing Arguments--")
    parser = init_parser()
    args = parser.parse_args()

    args.bidder_id = int(1)
    args.dataset = "gsvm"
    args.nbids = int(25000)

    #os.environ['WANDB_SILENT'] = "true"
    wandb.init(project="mvnn")
    wandb.config.update(args)

    main(args)

