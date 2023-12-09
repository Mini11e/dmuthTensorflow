'''look at specific functions - awareness/ not regularly used functions
custom gradients - talk about gradient tape again
'''

# two ANNs, in need of talking to each other
# image -> ANN -> text 
# output: propability vector (this represents word token vector) - take sample
# this output -> next ANN -> "is this beach yes no?"

# problem: can't differentiate the sample to train robot 1
# there is a "gamble_softmax" that can approximate the gradient of a sample function


# 2 goals: how to cite using latex
# implement custom gradient

''' write essay about code (4-10 pages), explaining the outcomes/
 citations/ manage citations - and why you want a citation manager
'''

# we want to cite paper 
# overleaf -> online Latex editor
# you can get of google scholar a bibteX something

# \cite()

# citation manager
# zotero or mandaleaf
    # extension for browser - import them in manager
    # export to use


import tensorflow as tf

sample_input = tf.random.uniform(shape=(4,2), minval = -3, maxval = 3)
# https://www.tensorflow.org/api_docs/python/tf/custom_gradient

@tf.custom_gradient # thing that you add above a function that adds some functionality
def log1pexp(x):
    e = tf.exp(x) # forward step

    def grad(upstream): # got a single input - the input is the leibnitz (error signal that we get from until then)
     # upstream is the gradient until where we are 
        return upstream * (1 - 1 / (1 + e))
  
    return tf.math.log(1 + e), grad # calculation - the second thing is the reference to the function above

@tf.custom_gradient
def my_sigmoid(x):
    sig_res = 1/(1+tf.exp(-x))
  
    # because we are smarter than tensorflow we write the backwards step
    # define the gradient function:
    def grad(upstream):
        # since sig_res is within scope: 
        diag_of_jacobian = sig_res * (1 - sig_res)
        # but we need to multiply it with the upstream
        return upstream * diag_of_jacobian

    return sig_res, grad

print(my_sigmoid(sample_input))


@tf.custom_gradient
def relu(x):

    relu_res = tf.where(x > 0, x, tf.zeros_like(x)) # zeros_like creates a tensor of 0 of shape x
    # relu_res = tf.where(x >= 0, x, 0) # not broadcastet, (my solution)
    # or 
    # relu_res = tf.max(0, x)

    def grad(upstream):

        gradient = tf.where(x > 0, tf.ones_like(x), tf.zeros_like(x))
        #gradient = tf.where(x >= 0, 1, 0) # also not broadcastet (my solution)

        return upstream * gradient
    
    return relu_res, grad


sample_input = tf.Variable(initial_value = sample_input) # we can now use them as weights, yaaay

'''with tf.GradientTape() as tape:
    tf_relu_act = tf.nn.relu(sample_input)
    my_relu_act = relu(sample_input)

grads_tf_relu = tape.gradient(tf_relu_act, [sample_input]) # this doesn't work - since grdaient tape stores all the gradients + calculations and just
                                                            #frees the memory after the first call of tape.gradient
grads_own_relu = tape.gradient(my_relu_act, [sample_input])'''


with tf.GradientTape(persistent = True) as tape: # now you actively need to free the memory, since it would just hog memory
    tf_relu_act = tf.nn.relu(sample_input)
    my_relu_act = relu(sample_input)

grads_tf_relu = tape.gradient(tf_relu_act, [sample_input])
grads_own_relu = tape.gradient(my_relu_act, [sample_input])
del tape # delete tape - or try to :)

print(grads_own_relu)
print(grads_tf_relu)


res = []
sample_input = tf.Variable(initial_value = tf.random.uniform(shape = (4,3)))

# no python inside this (no print etc)
@tf.function # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! speedup up to 20 times
def some_func(): 
    inputs = sample_input[step,:,:]
    for step in tf.range(10): # for loops over things that are tensorflow objects (or that can be transferred) 
                                # same goes for if-else functions
        with tf.GradientTape() as tape: 
            tf_relu_act = tf.nn.relu(sample_input)

    grads_tf_relu = tape.gradient(tf_relu_act, [sample_input])
    res.append(grads_tf_relu) # bad!


# tensorflow tensor array # use that 
