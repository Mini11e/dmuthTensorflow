import func
import mlp_layer as ml

class MLP:

    def __init__(self, num_layer = 4, layer_sizes = (8,8,8,8), learning_rate = 0.03):
        self.num_layer = num_layer

        assert len(layer_sizes) == num_layer, "The function got passed an amount of layer-sizes unequal to the one specified."
        assert type(layer_sizes) == tuple, f"The function did expect a tuple, got {type(layer_sizes)} instead"

        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate

        # creating a list to save all the MLP_Layers in there
        self.layer = []
        
        # populating said list with the number of neurons specified. 
        for i in range(self.num_layer):
            if i == 0:
                self.layer.append(ml.MLP_LAYER(in_size = 64, units = self.layer_sizes[i]))

            elif i == len(self.layer_sizes):
                # for this specific MLP the last layer NEEDS to have 10 neurons, so we just disregard the last entry of the tuple passed to the init
                self.layer.append(ml.MLP_LAYER(in_size = self.layer_sizes[i-1], units = 10, activ_func = func.softmax, func_deriv = func.cce_softmax_derivative))

            else:
                self.layer.append(ml.MLP_LAYER(in_size = self.layer_sizes[i-1], units = self.layer_sizes[i]))

    # function if one ever would like to print the MLP
    def __repr__(self):
        return f"This Neuronal Network has {self.num_layer} Layers, with {self.layer_sizes} Neurons respectively."


    # function for backpropagation over all layers
    def backwards(self, target):

        # first we need the error from the target/last layer
        error0 = func.cce_softmax_derivative(self.layer[-1].activation, target)
        error = "" # and a variable defined out of a loop

        # go over the layers (reversed)
        for layer in reversed(self.layer):

            if layer == self.layer[-1]:

                # calculate weight gradient and adjust weights
                layer.weights_backward(error0)
                layer.update_weights(learning_rate = self.learning_rate)

                # update error 
                error = layer.cross_backward(error0)

            
            else:

                # complete error calculation in layer
                error = layer.calc_error(error)

                # caculate weight gradiend and adjust weights with now fully calculated error
                layer.weights_backward(error)
                layer.update_weights(learning_rate = self.learning_rate)

                # and partial calulation for next leayer
                error = layer.cross_backward(error)


    # function for applying the MLP on the input  
    def forwards(self, input):

        for layer in self.layer:
            input = layer.forward(input)

        return input