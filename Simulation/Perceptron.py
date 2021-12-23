import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
# import matplotlib.animation as animation


class Perceptron:
	"Perceptron class"

	def __init__(self, N, learning_rate = 0.1):
		self.N = N
		self.weights = np.random.rand(N)*2-1
		self.learning_rate = learning_rate

	def predict(self, feature_tuple):
		"returns an output guess from inputs"
		return np.sign(np.sum(self.weights * feature_tuple))

	def predict_all(self, feature_tuples):
		"fraction of correct predictions from several inputs"
		predictions = np.zeros(len(feature_tuples))
		for i,features in enumerate(feature_tuples):
			predictions[i] = self.predict(features)
		return predictions

	def accuracy(self, feature_tuples, labels):
		"fraction of correct predictions from several inputs"
		predictions = np.zeros(len(feature_tuples))
		for i,features in enumerate(feature_tuples):
			predictions[i] = self.predict(features)
		return sum(predictions==labels)/len(labels)

	def train(self, feature_tuple, label):
		error = label - self.predict(feature_tuple)
		for i in range(self.N):
			self.weights[i] += error * feature_tuple[i] * self.learning_rate


if __name__ == "__main__":
	# generate some random features
	N = 500
	feature1 = np.random.rand(N)
	feature2 = np.random.rand(N)
	feature3 = np.ones(N) #  add 3rd feature (always 1) to simulate a trainable threshold
	labels = feature1 > feature2
	labels = labels.astype(int)*2-1
	features = [i for i in zip(feature1,feature2,feature3)]
	p = Perceptron(N=3)
	print(p.accuracy(features,labels))

	for i in range(100):
		p.train(features[i],labels[i])
	print(p.accuracy(features,labels))

	p.learning_rate = 0.005
	for i in range(N):
		p.train(features[i],labels[i])
	print(p.accuracy(features,labels))
	print(p.weights)

	#cache labels and weights
	cache_labels = []
	cache_weights = []
	p = Perceptron(N=3, learning_rate=0.05) # initialize p with random weights ('forget' previous training)
	for i in range(N):
		p.train(features[i],labels[i])
		cache_labels.append(p.predict_all(features))
		cache_weights.append(list(p.weights))

	# interactive plotting
	#plt.ion()
	# Create new Figure with black background
	fig = plt.figure(figsize=(8, 8), facecolor='black')
	# Add a subplot with no frame
	ax = plt.subplot(111, frameon=False)
	# No ticks
	ax.set_xticks([])
	ax.set_yticks([])
	predicted_labels = p.predict_all(features)
	scat = ax.scatter(feature1, feature2, c=predicted_labels)
	txt = ax.text(0, 1.0, cache_weights[0], color="w")

	for i in range(N):
		#p.train(features[i],labels[i])
		#scat.set_array(p.predict_all(features))
		scat.set_array(cache_labels[i])
		txt.set_text(cache_weights[i])
		plt.pause(0.1)

	print(p.accuracy(features,labels))
	print(p.weights)
	#plt.waitforbuttonpress()

	# optimal weights are [1,-1]:
	p.weights = np.array([1,-1,0])
	print(p.accuracy(features,labels))

	# how can the learning be improved? Change rate at some point?
	# use independent training and testing sets!
	# does it help to make several passes over the training data?
