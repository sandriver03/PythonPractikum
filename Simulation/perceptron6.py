import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, 5)
		self.conv2 = nn.Conv2d(10, 20, 5)
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, 10)

	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2(x), 2))
		x = x.view(-1, 20 * 4 * 4)  # Reshape to vector
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)

# Setup input transformation
batch_size = 128
transformation = transforms.Compose([
					   transforms.ToTensor(),
					   transforms.Normalize((0.1307,), (0.3081,))  # Standardization
				   ])

# Setup data loader
train_loader = torch.utils.data.DataLoader(
	datasets.MNIST('~', train=True, download=True, transform=transformation),
	batch_size=batch_size
)

device = torch.device('cpu')
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.5) # gradient descent

def train_epoch(model, device, train_loader, optimizer):
	# Set network to training mode
	model.train()
	# Iterate over dataset
	losses = list()
	for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
		# Move data to GPU
		data, target = data.to(device), target.to(device)
		# Clear gradients
		optimizer.zero_grad()
		# Compute output
		output = model(data)
		# Compute crossentropy loss
		loss = F.nll_loss(output, target)
		# Compute gradient
		loss.backward()
		# Perform gradient descent
		optimizer.step()
		# Track losses
		losses.append(loss.item())
	# Return loss at end of epoch
	return losses

losses = list()
for epoch in range(3):
	epoch_losses = train_epoch(model, device, train_loader, optimizer)
	print(f"Average loss in epoch {epoch}: {np.mean(epoch_losses):.5f}")
	losses.extend(epoch_losses)

plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Train loss")
plt.show()



# Now test how the model performs on new data
# Setup data loader
test_loader = torch.utils.data.DataLoader(
	datasets.MNIST('~', train=False, download=True, transform=transformation), batch_size=None)

# Set network to testing mode
model.eval()
corrects = list()
incorrect_images = list()
for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
	output = model(data[None,...]) 	# Compute output
	(max_vals, arg_maxs) = torch.max(output.data, dim=1) # arg_maxs is tensor of indices [0, 1, 0,..]
	correct = torch.sum(target==arg_maxs)
	if not correct:
		incorrect_images.append(data.squeeze())
	#print(f'{batch_idx}: {correct.item()}')
	corrects.append(correct.item())
accuracy = (sum(corrects) * 100.0 / batch_idx)
print(f'Test accuracy: {accuracy:.5f}')
plt.imshow(incorrect_images[0], cmap='gray_r')
plt.show()
