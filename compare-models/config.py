
### Parameters for the models ###
### MVNN parameters ###
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

### UMNN parameters ###
umnn_parameters = {"mon_in": 10, "cond_in": 0, "hiddens": [10,10], "n_out": 1, "nb_steps": 50, "device": "cpu"}

### CERT parameters ###
# new version of cert
CERT_parameters = {"output_parameters": 1, "num_hidden_layers": 4, "hidden_nodes": 20}
# paper loss for classification
CERT_parameters.update({"classif_loss": "cross_entropy", "regression_loss": "mse"})
# initial learning rate 5e-3 decrease for large lambda
CERT_parameters.update({"lr": 5e-3})
# normalize input feature to be between 0 and 1
CERT_parameters.update({"normalize_input": True})




