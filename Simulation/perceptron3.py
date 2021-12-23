# requires pytorch, to install call:
# conda install pytorch torchvision -c pytorch

# Perceptron using automatic gradient descent algorithm based on pytorch
import numpy as np
import torch
from torch.autograd import Variable

# Training data
x_data = torch.Tensor([1.0, 2.0, 3.0])
y_data = torch.Tensor([2.0, 4.0, 6.0])

test_data = 4.0 # test data

lr = 0.01  # learning rate
w = Variable(torch.Tensor([1.0]), requires_grad=True)

# forward pass of the model
def forward(x):
	return w * x

# loss function
loss_fn = torch.nn.MSELoss(reduction='sum') # 'mean' would also work

# Before training
print(f'predict (before training): {forward(test_data).item()}')

# defining an automatic optimizer
optimizer = torch.optim.SGD([w], lr=lr)

# Training loop
for _ in range(20):
	for x, y in zip(x_data, y_data):
		#1. forward
		y_hat = forward(x)
		#2. loss
		loss = loss_fn(y_hat, y)
		#3. backward
		loss.backward()
		#4. update
		optimizer.step()
		print(f'grad: {w.grad.data}, \t loss: {loss.item()}')
		# Manually zero the gradients after updating weights
		optimizer.zero_grad()

# After training
print(f'predict (after training): {forward(test_data).item()}')
