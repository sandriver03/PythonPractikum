import matplotlib.pyplot as plt
import numpy as np


# Perceptron using manual gradient descent algorithm
# what is the connections of this perceptron look like?
# Training data initialization
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# test data
test_data = 4.0

# learning rate
lr = 0.05
# a random guess, initial weight
w = 10.0


# forward pass of the model
def forward(w_vec, xs):
    # calculate the perceptron output from input and weights
    raise NotImplementedError


# loss function
def loss(y_hats, ys):
    # calculate the loss function (squared difference between intended output y and actual output y_hat)
    raise NotImplementedError


# compute gradient (backpropagation)
def backward(xs, ys, w_vec):
    # calculate the gradient of the loss function (derivative with respect to w, d_loss/d_w)
    raise NotImplementedError


if __name__ == '__main__':
    # Before training
    print(f'predict (before training): {forward(w, test_data)}')

    # Training loop
    for epoch in range(20):
        for x, y in zip(x_data, y_data):
            # 1. forward
            y_hat = forward(w, x)
            # 2. loss
            l = loss(y_hat, y)
            # 3. backward
            grad = backward(x, y, w)
            # 4. update: set the new value for w
            # TODO

        plt.figure(1)
        plt.plot(epoch, l, 'ro')

        plt.figure(2)
        plt.plot(np.mean(w), l, 'bo')
        print(f'grad: {round(grad, 4)}, \t loss: {round(l, 4)}')

    # After training
    print(f'predict (after training): {forward(w, test_data)}')
    plt.show()


# we briefly talked about the biasing unit. below demonstrates its functions
# this problem below cannot be accurately solved without the biasing unit
# x_data = [1., 2., 3.,]
# y_data = [2.5, 4.5, 6.5]
# now, add the biasing unit and try again
