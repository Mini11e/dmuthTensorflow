from imports import tf
from imports import np

class Purr(tf.keras.Model):
    def __init__(self, name = "purr"):

        super().__init__()

        inputs = tf.keras.Input(shape = (32, 32, 3), dtype = tf.float32)
    
        x = tf.keras.layers.Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = tf.nn.relu)(inputs)    # shape: [batch_size, 32, 32, 16]
        x = tf.keras.layers.Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = tf.nn.relu)(x)         # shape: [batch_size, 32, 32, 16]
        x = tf.keras.layers.MaxPooling2D()(x)                                                                           # shape: [batch_size, 16, 16, 16]

        for _ in tf.range(4):
            x = tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = tf.nn.relu)(x)     # shape: [batch_size, 16, 16, 32]
        x = tf.keras.layers.MaxPooling2D()(x)                                                                           # shape: [batch_size, 8, 8, 32]

        for _ in tf.range(4):
            x = tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = tf.nn.relu)(x)     # shape: [batch_size, 8, 8, 64]
        x = tf.keras.layers.GlobalAveragePooling2D()(x)                                                                 # shape: [batch_size, 64]

        outputs = tf.keras.layers.Dense(units = 10, activation = tf.nn.softmax)(x)

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