import tensorflow as tf


def create_monotonic_model(input_dim, num_hidden_layers, num_nodes):
    # Define the model inputs
    inputs = tf.keras.layers.Input(shape=(input_dim,))

    # Define the hidden layers
    x = inputs
    for _ in range(num_hidden_layers):
        x = tf.keras.layers.Dense(num_nodes, activation='relu')(x)

    # Define the output layer with monotonic activation
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    return model


# Example usage:
model = create_monotonic_model(input_dim=3, num_hidden_layers=2, num_nodes=10)




