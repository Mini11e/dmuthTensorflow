import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
  
import func
import mlp

# helper function to display digits
def plot_img(digits):

    plt.gray()
    plt.matshow(digits.reshape(8,8))
    plt.show()
    
# function to display loss over epochs
def plot_loss(loss):

    plt.plot(loss)
    plt.xlabel(xlabel = "Epochs")
    plt.ylabel(ylabel = "Average Loss")
    plt.show()

if __name__ == "__main__":

    data, target = load_digits(return_X_y = True) # roughly 1800 entries

    # adjust to values between [0:1] and cast to float32
    data = np.float32(data)/20 

    # make them one-hot vectors
    target = func.weird_vectors(target) 
    # shuffle 'em good
    data, target = func.pirate_shuffle(arr1 = data, arr2 = target, minib_size = 10) 
    
    # create MLP with arbitrary amount of layers/neurons. (64 for first layer, as it has to equal one dataset, 10 for last layer, as it has to equal target)
    MLP = mlp.MLP(num_layer = 4, layer_sizes = (64, 128, 64, 10))

    # train the network, save the loss
    epochs = 100
    loss = np.zeros((7, epochs))
    helper = 0
    
    for part_data, part_target  in zip(data[0:7], target[0:7]):

        loss[helper] = func.train(ann = MLP, input = part_data, target = part_target, epochs = epochs)
        helper += 1
        
        print(f'weiteres minibatch abgeschlossen') # our progress bar

    # test the network on a number it hasn't been trained on
    pred = MLP.forwards(data[-1][0])
    print(f'prediction: {np.argmax(pred)}')
    print(f'target:     {np.argmax(target[-1][0])}')

    plot_loss(loss = np.mean(loss, axis = 0))



    ###########################
    # we let it run 4 times, trained on batches 0-7, tested on batch 9:
    # 
    # prediction: 9
    # target: 9
    #
    # prediction: 3
    # target: 2
    #
    # prediction: 9
    # target: 9
    # 
    # prediction: 3
    # target: 3
    ###########################


    
