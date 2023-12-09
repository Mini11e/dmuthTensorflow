from imports import tf
import func
import meow

if __name__ == "__main__":
    LEARNING_RATE0 = 0.001
    LEARNING_RATE1 = 0.5

    BATCH_SIZE = 128
    NUM_EPOCHS = 25

    # memory management - delete one model after training

    #########################################################################
    # Architectures: Purr, Meow (block network, pyramid shaped network)     #
    # learning rates: 0.001 and 0.5                                         #
    # optimiser: Adam, SGD                                                  #
    #########################################################################

    # we save each model individually, so that we can free up the memory after training
    metrics = []
    names = ['Meow_ADAM_0.001', 'Meow_ADAM_0.5', 'Meow_SGD_0.001', 'Meow_SGD_0.5']
    train_ds, test_ds = func.load_and_prep_cifar(BATCH_SIZE)

    for i in range(4):
        model = meow.Meow()

        if i%2 == 0:
            lr = LEARNING_RATE0
        else:
            lr = LEARNING_RATE1
        
        if i < 2:
            model.set_optimiser(learning_rate = lr)
        else:
            model.set_optimiser(optimiser = tf.keras.optimizers.legacy.SGD(learning_rate = lr))

        model.set_metrics()
        model.set_loss_function()

        metrics.append(model.train_loop(train_ds, test_ds, NUM_EPOCHS))

        del model


    func.visualise(metrics, names)

    input("wait for user to continue")

    print(metrics)

