import numpy as np
import matplotlib.pyplot as plt


"""
perceptron as classifier
"""

# generate some random features
N_samples = 500
xs = np.random.rand(N_samples, 2)
y = xs[:, 0] > xs[:, 1]
y = y.astype(int)*2-1  # labels are -1 and 1
# how this data looks like?


# training in each trial will be the same as we saw last time, with 4 steps
# let's write everything in array or matrix form since it is very powerful (faster and better scalable)
# calculate perceptron output
# remember we are doing classification now. check the value of y. what is a good activation function to use?
def forward(x_vec, weights):
    """
    Forward pass of the model, returns an output guess from inputs (y_hat)
    :param x_vec:
    :param weights:
    :return:
    """
    raise NotImplementedError


# calculate loss (or the error of perceptron output vs. real data)
def loss(y_hats, y_vec):
    """
    Loss function
    :param y_hats:
    :param y_vec:
    :return:
    """
    raise NotImplementedError


# calculate gradient at current weights
def backward(x_vec, y_vec, y_hats, weights):
    """
    Computes gradient d_loss/d_w for each w (backpropagation)
    :param x_vec:
    :param y_vec:
    :param y_hats:
    :param weights:
    :return:
    """
    raise NotImplementedError


# finally update the weights
def update(weights, grad, learning_rate):
    """
    update the weights based on current gradient and learning rate
    :param weights:
    :param grad:
    :param learning_rate:
    :return:
    """
    raise NotImplementedError


# let's put all these 4 steps into one function
# remember we are doing this for 1 trial
def train(y_vec, x_vec, weights, learning_rate):
    """
    train the model
    :param y_vec:
    :param x_vec:
    :param weights:
    :param learning_rate:
    :return:
    """
    # forward
    y_hat = forward(x_vec, weights)
    # loss
    l = loss(y_hat, y_vec)
    # backwards
    grad = backward(x_vec, y, y_hat, weights)
    # update weights
    weights = update(weights, grad, learning_rate)
    return weights

# however, different from regression, this time we care about the model's ability to make predictions
# in other words, we need to get the prediction accuracy for our model
# let's randomly split the data into 2 parts, maybe 80% vs 20%, use the larger part of the data
# to train the model (the training set), and then test the accuracy of the trained model with the remaining
# 20% of the data (the validation set)
# in our example there is no noise, so this training-validation strategy is not so important. however if the size of
# data is small and/or there is noise in the data, then the strategy (or improved one called cross-validation) is
# very important to guard against over fitting


# a function to calculate accuracy, given xs from different trial
# note: you could write the function to work on one trial then apply it to many trials in the cross-validation
# data with a for loop, but you can also directly use np.array math - this is much faster compared to the for
# loop
def prediction_accuracy(x_vec, y_vec, weights):
    """
    calculate prediction accuracy of the model
    :param x_vec:
    :param y_vec:
    :param weights:
    :return:
    """
    raise NotImplementedError


# now let' setup the training and validation
# first: remember it is important to and the biasing unit (the constant 1s)
# TODO

# then, split the data in to training set and validation set
# TODO

# now, train the model with the training set
# remember first choose a learning rate
# randomly choose some initial weights
# TODO

# check the performance of the trained model with the validation set
# TODO


# how can the learning be improved? Change rate at some point?
# use independent training and testing sets!
# does it help to make several passes over the training data?




