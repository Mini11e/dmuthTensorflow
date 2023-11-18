import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
ds = tfds.load('mnist', split = 'train')

#pipeline
ds = ds.map(lambda feature_dict: (feature_dict['image'], feature_dict['label'])) #default load(as_supervised=False): comes as dict, load(as_supervised=True): tuple
ds = ds.map(lambda image, label: (tf.reshape(image, (-1)), label))
ds = ds.map(lambda image, label: ((tf.cast(image, tf.float32)/128)-1, label))
ds = ds.map(lambda image, label: (image, tf.one_hot(label, depth = 10)))
ds = ds.shuffle(1024).batch(256)
ds = ds.prefetch(4)

#create mlp
class MLP_Model(tf.keras.Model):
    def __init__(self, layer_sizes, output_size = 10):
        super().__init__()
        self.mlp_layer = []

        for layer_size in layer_sizes:
            new_layer = tf.keras.Dense(units = layer_size, activation =  'sigmoid')
            self.mlp_layers.append(new_layer)
        self.output_layer = tf.keras.layers.Dense(units = output_size, activation = 'softmax')

def call(self, x):
    for mlp_layer in self.mlp_layers:
        x = mlp_layer(x)
    y = self.output_layer(x)
    return y

#training
EPOCHS = 10

model = MLP_Model(layer_sizes = [256, 256])
cce = tf.keras.losses.CategoricalCrossentropy()

optimizer = tf.keras.optimizers.legacy.SGD()
ds = ds

losses = []

for epoch in range(EPOCHS):
    
    for x, target in ds:
        
        with tf.GradientTape() as tape:
            pred = model(x)
            loss = cce(target, pred)
        gradients = tape.gradient(loss, model.variables) #calculate outside of the GradientTape context
        optimizer.apply_gradients(zip(gradients, model.variables))
        losses.append(loss.np())
    print(np.mean(losses))




           