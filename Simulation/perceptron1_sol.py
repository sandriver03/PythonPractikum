import matplotlib.pyplot as plt
import numpy as np


# Perceptron using manual gradient descent algorithm
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
	return w_vec * xs


# loss function
def loss(y_hats, ys):
	# calculate the loss function (squared difference between intended output y and actual output y_hat)
	return (y_hats - ys) ** 2


# compute gradient (backpropagation)
def backward(xs, ys, w_vec):
	# calculate the gradient of the loss function (derivative with respect to w, d_loss/d_w)
	return (forward(w_vec, xs) - ys) * x


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
		# 4. update
		w += -lr * grad

	plt.figure(1)
	plt.plot(epoch, np.sum(l), 'ro')

	plt.figure(2)
	plt.plot(np.mean(w), np.sum(l), 'bo')
	print(f'grad: {round(grad,4)}, \t loss: {round(l,4)}')


# After training
print(f'predict (after training): {w, forward(w, test_data)}')
plt.show()
