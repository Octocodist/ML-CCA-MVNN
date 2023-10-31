import numpy
import pickle


from mvnns.mvnn import MVNN
from mvnns.mvnn_generic import MVNN_GENERIC

def load_dataset(filepath):
    # load dataset using pickle
    # parse filepath
    with open(filepath, "rb") as file:
        dataset = pickle.load(file)



    return dataset
def get_mvnn():

   model = MVNN_GENERIC(input_dim=X_train.shape[1],
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
                    init_little_const = init_little_const
                    ))
def main():
    print("--Start Program--")
    filepath = "./dataset_generation/datasets/mrvm/mrvm_1_1.pkl"
    dataset = load_dataset(filepath)
    print(dataset)

    #model = MVNN()





if __name__ == "__main__":
    main()

