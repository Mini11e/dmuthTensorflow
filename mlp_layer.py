import numpy as np
import func

class MLP_LAYER:

    def __init__(self, in_size = 32, units = 32, activ_func = func.sigmoid, func_deriv = func.sigmoid_derivative):
        self.units = units
        self.in_size = in_size
        self.activ_func = activ_func
        self.func_deriv = func_deriv

        self.bias = np.zeros(self.units)
        self.weights = np.random.normal(0, 0.2, (self.in_size, self.units)) # setting random weights as init

    # set bias, even though we never used it - in case we wanted to :)
    def set_bias(self, bias):
        
        assert np.shape(bias) == np.shape(self.bias), f"The function received a bias that isn't correct. Should have gotten an array of shape {np.shape(self.bias)}"
        self.bias = bias



    def forward(self, input):

        # for forwarding, the input should be of shape (neurons layer -1,)
        assert np.shape(input)[0] == self.in_size, "The function got passed a shape of inputs contradicting the setup of the layer."

        self.input = input
        # we saved preactivation and activation in the layer itself instead of a dictionary outside - this is more intuitive to us
        self.preactivation = (input @ self.weights) + self.bias
        self.activation = self.activ_func(self.preactivation)

        # triple checking everything went well
        assert np.shape(self.activation)[0] == self.units, "Please check all inputs again, something went wrong."

        return self.activation

    # calculate the gradient for the weights
    def weights_backward(self, error_signal):

        # and we also saved the gradients in here, for consistency sake.
        self.weight_gradient = np.outer(error_signal, self.input)


    # calculating part of the error for the next layer ()
    def cross_backward(self, error_part):

        derivativeLinput = self.weights @ error_part

        return derivativeLinput
    
    # caculate the error, given the partial error from the previous layer
    def calc_error(self, error_prev): # error_prev should be derivativeLinput of previous layer

        error = error_prev * self.func_deriv(self.preactivation)
        
        # and that should have the shape of (num_units,)
        assert error.shape[0] == self.units

        return error

    # apply the gradient to the weights
    def update_weights(self, learning_rate):
        self.weights = self.weights - learning_rate * self.weight_gradient.T
