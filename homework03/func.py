from imports import tf
from imports import tfds
from imports import plt
from imports import np

def load_and_prep_cifar(batch_size):
    train, test = tfds.load('cifar10', split = ['train', 'test'], as_supervised = True, shuffle_files = True)

    def preprocessing(img, label):
        img = tf.cast(img, tf.float32)
        img = (img/128) - 1
        label = tf.one_hot(label, depth = 10)
        return img, label
    
    train = train.map(lambda img, label: preprocessing(img, label), num_parallel_calls = tf.data.AUTOTUNE)
    test  =  test.map(lambda img, label: preprocessing(img, label), num_parallel_calls = tf.data.AUTOTUNE)

    train = train.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test  =  test.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train, test


def visualise(data, names):

    if len(data) != len(names): raise ValueError('You passed an amount of Names that doesn\'t match the amount of data')

    titles = ['Train Loss', 'Train Acc', 'Test Loss', 'Test Acc']
    fig, axes = plt.subplots(nrows = len(data), ncols = 4)
    
    for i in range(len(data)):
        for j in range(4):
            axes[i][j].plot(data[i][j])
            
            # setting the column headers
            if i == 0: 
                axes[i][j].set_title(titles[j])
        
        # setting row labels
        axes[i][0].set_ylabel(names[i])

    plt.show()
