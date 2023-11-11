import numpy as np

# functions and their derivatives
def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 -sigmoid(x))


def softmax(x):
    ep = np.exp(x)
    return ep / np.sum(ep)


def cce(x, target):
    assert x.all() >= 0 and x.all() <= 1, f"x is not between 0 and 1"

    return -np.sum(target * np.log(x))

def cce_softmax_derivative(x, target):
    return x - target


# function for making target one-hot vectors
def weird_vectors(target):

    vectors = np.zeros((len(target), 10), dtype = np.int32)
    
    helper = 0 # bcs we are too lazy to write two for loops 
    for i in target:
        vectors[helper][i] = 1
        helper += 1

    return vectors


# function for shuffling and minibatching (pirates because of arr, you know? arr.)
def pirate_shuffle(arr1, arr2, minib_size = 0):

    # since we have to shuffle the arrays (data, target) in parallel, they need to have the same length.
    assert len(arr1) == len(arr2), "array 1 and 2 should have the same length!"

    # created a permutation that can later be used for indexing (shuffling)
    perm = np.random.permutation(len(arr1))

    if(minib_size != 0):

        # if minibatches got specified, split the whole data in corresponding amounts
        arr1 = np.array_split(arr1[perm], minib_size)
        arr2 = np.array_split(arr2[perm], minib_size)

        return arr1, arr2

    else:

        return arr1[perm], arr2[perm]
    
    
# training the mlp and calulating the mean loss for each epoch
def train(ann, input, target, epochs = 5):
    
    mean_loss = np.zeros(len(input), dtype = np.float32)
    total_mean = np.zeros(epochs, dtype = np.float32)

    for epoch in range(epochs):
        for i in range(len(input)):
            a = ann.forwards(input[i])
            ann.backwards(target[i])
            mean_loss[i] = cce(a, target[i])

        total_mean[epoch] = np.mean(mean_loss)
    
    return total_mean
