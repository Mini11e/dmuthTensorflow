from imports import tf
from imports import np

class Meow(tf.keras.Model):
    def __init__(self, name = "meow"):
        
        super().__init__()

        inputs = tf.keras.Input(shape = (32, 32, 3), dtype = tf.float32)

        x = inputs
        for i in tf.range(9):  
            ########################
            # that would result in the first layer having 2 filters and the last layer 32768 filters - do we really want to do this? 
            # or just cut off after 11 (range 12) steps, 4069 filter
            #######################


            # since we're reducing the image size with padding = 'valid' and kernel_size = 3 by exactly 1 pixel in each direction (2 in total), 
            # we have to run it through 16 layers to be at an image size of 0. But since one can't calculate with that, we run only 11 layers
            # on it, and in the last layer global-pool-average it to reduce the 2x2 image to a vector 
            x = tf.keras.layers.Conv2D(filters = 2**(i+1), kernel_size = 3, padding = 'valid', activation = tf.nn.relu)(x)   # shape: [batch_size, 30, 30, (powers of 2)] 
            # or start with i+3 and stop at step 9 (range(10))


        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        outputs = tf.keras.layers.Dense(10, activation = tf.nn.softmax)(x)

        self.model = tf.keras.Model(inputs = inputs, outputs = outputs, name = name)

    def __call__(self, x):
        self.call(x)

    @tf.function
    def call(self, x):
        self.model(x)

    def set_loss_function(self, loss_function = tf.keras.losses.CategoricalCrossentropy()):
        self.loss_function = loss_function

    def set_metrics(self, loss_metric = tf.keras.metrics.Mean(name = 'loss'), accuracy_metric = tf.keras.metrics.CategoricalAccuracy(name = 'acc')):
        self.loss_metric = loss_metric
        self.accuracy_metric = accuracy_metric 

    def get_metrics(self):
        return self.loss_metric.result(), self.accuracy_metric.result()
  
    def reset_metrics(self): 
        self.loss_metric.reset_states()
        self.accuracy_metric.reset_states()

    def set_optimiser(self, optimiser = None, learning_rate = 0.001):
        '''
        Sets the Optimiser for the model. 
            If optimiser = None (default), tf.keras.optimizers.Adam() is being used. learning_rate defaults to 0.001.
        '''
        if optimiser == None:
            self.optimiser = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        else:
            self.optimiser = optimiser

    def write_to_file(self, path_to_file, filename):
        pass

    @tf.function
    def train_step(self, data):
        
        x, target = data
    
        with tf.GradientTape() as tape:
            pred = self.model(x)
            loss = self.loss_function(target, pred)

        self.loss_metric.update_state(loss)
        self.accuracy_metric.update_state(target, pred)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimiser.apply_gradients(zip(gradients, self.model.trainable_variables))


    @tf.function
    def test_step(self, data):

        x, target = data

        pred = self.model(x)
        loss = self.loss_function(target, pred)

        self.loss_metric.update_state(loss)
        self.accuracy_metric.update_state(target, pred)


    def train_loop(self, train, test, num_epochs):
        
        metrics = np.empty((4, num_epochs))

        for epoch in range(num_epochs):

            print(f'Epoch {epoch}')
            for data in train:
                self.train_step(data)

            metrics[0][epoch], metrics[1][epoch] = self.get_metrics()
            self.reset_metrics()

            print(f'Training Loss: {metrics[0][epoch]}, Training Accuracy: {metrics[1][epoch]}')

            for data in test:
                self.test_step(data)

            metrics[2][epoch], metrics[3][epoch] = self.get_metrics()
            self.reset_metrics()

            print(f'Test Loss: {metrics[2][epoch]}, Test Accuracy: {metrics[3][epoch]}')


        return metrics