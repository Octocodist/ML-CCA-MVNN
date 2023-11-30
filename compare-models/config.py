#Config File for the models
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

cert_parameters = {"output_parameters": 1, "num_hidden_layers": 4, "hidden_nodes": 20}


