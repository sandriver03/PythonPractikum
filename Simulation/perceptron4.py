# Multilayer Perceptron using automatic gradient descent algorithm based on pytorch
import matplotlib.pyplot as plt
import torch
import torch.autograd

device = torch.device('cpu') # 'cuda' would run it on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N = 64 # training batch size
D_in = 1000 # input layer size
H1 = 1000 # hidden layer size
H2 = 1000 # second hidden layer size
# define another layer size here!
D_out = 10 # output layer size

# Create random Tensors to hold inputs and outputs
x_data = torch.randn(N, D_in, device=device)
y_data = torch.randn(N, D_out, device=device)

lr = 0.001  # learning rate

#Step 1: Design our NN model class in pytorch way
class Model(torch.nn.Module):

	def __init__(self):
		super(Model, self).__init__()
		self.linear = torch.nn.Sequential( #sequential container
				torch.nn.Linear(D_in, H1),
				torch.nn.ReLU(),
				torch.nn.Linear(H1, H2),
				torch.nn.ReLU(),
				# add another layer here!
				torch.nn.Linear(H2, D_out)
			).to(device)

	def forward(self, x):
		return self.linear(x)

# our model
model = Model()

# Step 2: Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the nn.Linear layers.
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

losses = list()
# Step 3: Training cycle: forward, loss, backward, step
for epoch in range(100):
	# 1. forward
	y_hat = model.forward(x_data)
	# 2. loss
	loss = loss_fn(y_hat, y_data)
	# 2-1 zero gradients
	optimizer.zero_grad()
	# 3. backward
	loss.backward(retain_graph=True)
	# 4. update
	optimizer.step()
	print(f'loss: {loss.item()}')
	# Track losses
	losses.append(loss.item())

plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Training loss")
plt.show()
