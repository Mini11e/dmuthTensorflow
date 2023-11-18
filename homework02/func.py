import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import mlp


# visualisation for tracking accuracy while training, after training and loss while training and after training
def visualise(acc_epoch, acc_train, loss_epoch, loss_train):
    fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 2)
    
    ax1[0].plot(acc_epoch)
    ax1[0].axhline(1, alpha = 0.5, color = "red", linestyle = "--")
    ax1[0].set_xlabel('epochs')
    ax1[0].set_ylabel('accuracy')
    
    ax1[1].plot(acc_train)
    ax1[1].axhline(1, alpha = 0.5, color = "red", linestyle = "--")
    ax1[1].set_xlabel('every 20th image')
    ax1[1].set_ylabel('accuracy')
    
    ax1[0].set_ylim(0,1.3)
    ax1[1].sharey(ax1[0])


    ax2[0].plot(loss_epoch)
    ax2[0].set_xlabel('epochs')
    ax2[0].set_ylabel('loss')
    ax2[0].sharex(ax1[0])

    ax2[1].plot(loss_train)
    ax2[1].set_xlabel('every 20th image')
    ax2[1].set_ylabel('loss')
    ax2[1].sharex(ax1[1])

    ax2[1].set_ylim(0, max(np.max(loss_epoch), np.max(loss_train)) + 0.5)
    ax2[0].sharey(ax2[1])


    fig.tight_layout()

    plt.show()

# visualisation for every variation of the ANN and othe parameters we tried
def vis_accs(accs):

    cols = ['Base', 'Change 1', 'Change 2']
    # rows depicts our variation variables
    rows = ['Learning Rate', 'Batchsize', 'No. Layers', 'No. Neurons/Layer', 'Optimiser']
    # titles specifies the actual changes
    titles = [[0.1, 0.03, 0.5], [128, 32, 512], [2, 1, 8], [256, 64, 512], ['SGD', 'Lion', 'Adam']]


    fig, axes = plt.subplots(nrows = 5, ncols = 3)
    fig.tight_layout()  

    for i in range(5):
        for j in range(3):
            axes[i][j].plot(accs[i][j])
            axes[i][j].axhline(1, alpha = 0.5, color = "red", linestyle = "--")
            axes[i][j].set_ylim(0, 1.2)
            axes[i][j].set_xlabel('Epochs')
            axes[i][j].set_ylabel('Accuracy')
            axes[i][j].set_title(titles[i][j])

    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy = (0.5, 1.2), xytext=(0, 5),
                xycoords = 'axes fraction', textcoords = 'offset points',
                size = 'medium', ha = 'center', va = 'baseline')

    for ax, row in zip(axes[:,0], rows):
        ax.annotate(row, xy = (0, 0.5), xytext = (- ax.yaxis.labelpad - 5, 0),
                xycoords = ax.yaxis.label, textcoords = 'offset points',
                size = 'medium', rotation = 90, ha = 'right', va = 'center')
    
    fig.set_size_inches(10, 30)
    plt.show()


# pipelline for preparing data
def pipeline(ds, batch_size = 128):
    ds = ds.map(lambda image, label: ((tf.cast(image, tf.float32)/128)-1, label))
    ds = ds.map(lambda image, label: (tf.reshape(image, (-1,)), label))
    ds = ds.map(lambda image, label: (image, tf.one_hot(label, depth = 10)))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(16)
    return ds

# training function for tracking all accuracies and losses
def training_track_all_params(model, 
             train,
             test,
             optimiser,
             loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits = True),
             epochs = 10):
    
    # helpers and arrays for saving accuracies and losses
    step_size = 20
    acc_test = np.empty(epochs)
    acc_self = np.empty((epochs, int(len(train)/step_size + 1)))
    loss_test = np.empty(epochs)
    loss_self = np.empty((epochs, int(len(train)/step_size + 1)))
    
    # the actual training
    for epoch in range(epochs):

        counter = 0

        # iterate total data (x is now minibatch)
        for x, target in train:
            
            with tf.GradientTape() as tape: 
                pred = model(x)
                loss = loss_func(target, pred)

            # calculate gradients and apply
            gradients = tape.gradient(loss, model.variables)
            optimiser.apply_gradients(zip(gradients, model.variables)) 
            
            #track accuracy during training
            if counter % step_size == 0: 
                temp = tf.nn.softmax(pred)
                acc_self[epoch, int(counter/step_size)] = np.mean(np.argmax(temp, -1) == np.argmax(target, -1))
                loss_self[epoch, int(counter/step_size)] = np.mean(loss)
            counter += 1

        # accuracy after training using the test dataset
        acc_test[epoch], loss_test[epoch] = testing(model, test, loss_func)

        # basically a progsess bar
        print(f'Epoch {epoch}: with an accuracy of {round(acc_test[epoch], ndigits = 4)} and loss of {round(loss_test[epoch], ndigits = 4)}')
  
    return np.mean(acc_self, axis = 0), acc_test, np.mean(loss_self, axis = 0), loss_test

# training function for tracking just the accuracy after training (used for parameter variations)
def training_track_acc_epoch(model, 
             train,
             test,
             optimiser,
             loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits = True),
             epochs = 5):
    
    # array to save accuracy over epoch
    acc_test = np.empty(epochs)
    
    # actual training
    for epoch in range(epochs):
        for x, target in train:
            
            with tf.GradientTape() as tape: #track accuracy within training
                pred = model(x)
                loss = loss_func(target, pred)

            gradients = tape.gradient(loss, model.variables) #calculate outside of the GradientTape context
            optimiser.apply_gradients(zip(gradients, model.variables)) 

        # save accuracy
        acc_test[epoch], _ = testing(model, test, loss_func)

        # "progress bar"
        print(f'Epoch {epoch}: with an accuracy of {round(acc_test[epoch], ndigits = 4)}')
     
    return acc_test


# validation of the model using the test dataset
def testing(model, test, loss_func):
    
    # helpers and arrays for accuracy and loss
    accuracy = np.empty(len(test))
    loss = np.empty(len(test))
    i = 0

    # iterating over the test dataset
    for x, target in test:
        pred = model(x)
        pred = tf.nn.softmax(pred)

        # saving the loss/accuracy
        loss[i] = np.mean(loss_func(target, pred))
        accuracy[i] = np.mean(np.argmax(pred, -1) == np.argmax(target, -1))

        i += 1
    
    # returning the accuracy and loss
    return np.mean(accuracy), np.mean(loss) - 1



########################################################################################
# Variations of the ANN
########################################################################################

# variating the learning rate, all other parameters default
def var_learn(learning_rate, train, test, ann):

    optimiser = tf.keras.optimizers.legacy.SGD(learning_rate = learning_rate)

    return training_track_acc_epoch(ann, train, test, optimiser)


# variating the batch size, all other parameters default
def var_batch(batch, train_ds, test_ds, ann):

    train = pipeline(train_ds, batch)
    test = pipeline(test_ds, batch)

    optimiser = tf.keras.optimizers.legacy.SGD(learning_rate = 0.1)
    return training_track_acc_epoch(ann, train, test, optimiser)


# variating the layer count, all other parameters default
def var_net(layer, train, test, optimiser):

    ann = mlp.MLP_Model([128] * layer)
    return training_track_acc_epoch(ann, train, test, optimiser)


# variating the neurons per layer, all other parameters default
def var_layer(neurons, train, test, optimiser):

    ann = mlp.MLP_Model([neurons] * 2)
    return training_track_acc_epoch(ann, train, test, optimiser)


# variating the optimisers, all other parameters default
def var_opt(opt, ann, train, test):

    return training_track_acc_epoch(ann, train, test, opt)
    

