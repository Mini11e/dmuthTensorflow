import tensorflow as tf

class MLP_Model(tf.keras.Model):
    def __init__(self, layer_sizes, output_size = 10):
        super().__init__()
        self.mlp_layers = []
        
        # create layers according to layer_sizes e.g. [128, 256, ...]
        for layer_size in layer_sizes:
            self.mlp_layers.append(tf.keras.layers.Dense(units = layer_size, activation = 'sigmoid'))
        
        self.output_layer = tf.keras.layers.Dense(units = output_size)

    # call is our forward function
    def call(self, x):
        
        for layer in self.mlp_layers:
            x = layer(x)
        y = self.output_layer(x)
        
        return y