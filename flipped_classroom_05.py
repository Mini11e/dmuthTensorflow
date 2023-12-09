import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

ds = tfds.load('fashion_mnist', split =['train', 'test'], as_supervised = True, with_info = True, shuffle_files = True)

batch_size = 10

ds = ds.map(lambda image, label: (tf.reshape(image, (-1,)), label)) # flatten image
ds = ds.map(lambda image, label: ((tf.cast(image, tf.float32)/128)-1, label)) # normalise between -1 and 1, float better for calculation
ds = ds.map(lambda image, label: (image, tf.one_hot(label, depth = 10))) # make one hot vectors
ds = ds.shuffle(1000).batch(batch_size).prefetch(2) # shuffle locally not globally, batch to calculated gradient for multiple datapoints at a time

##### long version to create a model
class MLP_Model(tf.keras.Model):
    def __init__(self, layer_sizes, output_size = 10):
        super().__init__()
        self.mlp_layers = []
        
        # create layers according to layer_sizes e.g. [128, 256, ...]
        for layer_size in layer_sizes:
            self.mlp_layers.append(tf.keras.layers.Dense(units = layer_size, activation = 'relu'))
        
        self.output_layer = tf.keras.layers.Dense(units = output_size, activation = 'softmax')

    # forward function
    def call(self, x, training = True):
        
        for layer in self.mlp_layers:
            x = layer(x)
        y = self.output_layer(x)
        
        return y
    
ann = MLP_Model(layer_sizes = [256, 256])
optimiser = tf.keras.optimizers.Adam(learning_rate = 0.1)
epochs = 20


for epoch in range(epochs):

        for x, target in ds:
            
            with tf.GradientTape() as tape: 
                pred = ann(x)
                loss = tf.cce(target, pred)

            # calculate gradients and apply
            gradients = tape.gradient(loss, ann.variables)
            optimiser.apply_gradients(zip(gradients, ann.variables)) 

##### shorter version to create a model, but don't use it (terrible to debug)
layer_sizes = []
ann2 = tf.keras.Sequential()
ann2.add(tf.keras.Input(shape= (784, ), dtype = tf.float32))

for layer_size in layer_sizes:
    ann2.add(tf.keras.layers.Dense(layer_size, activation = 'relu'))
ann2.add(tf.keras.layers.Dense(layer_size, activation = 'softmax'))

##### even shorter version to create a model, but don't use it (terrible to debug)
ann3 = tf.keras.Sequential([tf.keras.layers.Dense(layer_size,  activation = 'relu') for layer_size in layer_sizes] + [tf.keras.layers.Dense(10, activation = 'softmax')])


##### rather use Functional API "keras.Model"
inputs = tf.keras.Input(shape = (784, ), dtype = tf.float32) #good for debugging because it makes sure the input is correct
x = inputs
for layer_size in layer_sizes:
    x = tf.keras.layers.Dense(layer_size, activation = 'relu')(x)
y = tf.keras.layers.Dense(10, activation = 'softmax')(x)

ann4 = tf.keras.Model(inputs = inputs, outputs = y, name = 'Ann4')