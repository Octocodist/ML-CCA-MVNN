import numpy
import pickle

import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from torch.optim import Adam
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset



from mvnns.mvnn import MVNN
from mvnns.mvnn_generic import MVNN_GENERIC
def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset to use", default="lsvm")
    parser.add_argument("--nbids", help="number of bids to use", default=20)
    parser.add_argument("--bidder_id", help="bidder id to use", default=3)
    parser.add_argument("--model", help="model to use", default="mvnn")
    return parser
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

    """if 'GSVM' in path:
            world = 'GSVM'
            N = 7
            M = 18
        elif 'LSVM' in path:
            world = 'LSVM'
            N = 6
            M = 18
        elif 'SRVM' in path:
            world = 'SRVM'
            N = 7
            M = 29
        elif 'MRVM' in path:
            world = 'MRVM'
            N = 10
            M = 98 
            
            SATS_parameters['SATS_domain'] in ['LSVM', 'GSVM']:
            self.generic_domain = False
            self.good_capacities = np.array([1 for _ in range(self.M)])
        elif SATS_parameters['SATS_domain'] in ['MRVM', 'SRVM']:
            self.generic_domain = True
            capacities_dictionary = SATS_auction_instance.get_capacities()
            self.good_capacities = np.array([capacities_dictionary[good_id] for good_id in self.good_ids])
        """
    X = dataset[0]
    y = dataset[1]
    if train_percent == 0:
        train_percent = len(X)/num_train_data
    X_train, test_and_val_X, y_train, test_and_val_y = train_test_split(X, y, test_size=train_percent, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(test_and_val_X,test_and_val_y, test_size=0.5, random_state=1)

    # transform to tensors
    X_train_tensor = torch.FloatTensor(X_train).float()
    y_train_tensor = torch.FloatTensor(y_train).float()
    X_val_tensor = torch.FloatTensor(X_val).float()
    y_val_tensor = torch.FloatTensor(y_val).float()
    X_test_tensor = torch.FloatTensor(X_test).float()
    y_test_tensor = torch.FloatTensor(y_test).float()

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
    train, val, test = load_dataset(args, train_percent=0.1)

    model = get_mvnn(train[0][0].shape[0])
    loss_mse = torch.nn.MSELoss()

    #model.print_parameters()
    batch_size = 12

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    epochs = 1
    optimizer = Adam(model.parameters())

    ### Training ###
    for e in tqdm(range(epochs)):
        for batch in train_loader:
            optimizer.zero_grad()

            predictions = model.forward(batch[0])

            loss = loss_mse(predictions,batch[1][:,1])

            loss.backward()
            model.transform_weights()
            optimizer.step()

    ### Validation ###
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True)
    for batch in val_loader:
        predictions = model.forward(batch[0])
        loss = loss_mse(predictions,batch[1][:,1])
        print(loss.item())

    ### Test ###
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)
    for batch in test_loader:
        predictions = model.forward(batch[0])
        loss = loss_mse(predictions,batch[1][:,1])
        print(loss.item())




if __name__ == "__main__":
    print("--Start Parsing Arguments--")
    parser = init_parser()
    args = parser.parse_args()
    main(args)

