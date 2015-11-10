'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np

class NeuralNet:

    def __init__(self, layers, epsilon=0.12, learningRate=0.3, numEpochs=100):
        '''
        Constructor
        Arguments:
        	layers - a numpy array of L-2 integers (L is # layers in the network)
        	epsilon - one half the interval around zero for setting the initial weights
        	learningRate - the learning rate for backpropagation
        	numEpochs - the number of epochs to run during training
        '''
	self.layers = layers
	self.epsilon = epsilon
	self.learningRate = learningRate
	self.numEpochs = numEpochs
	self.neuralNet = None
	self.weightThetas = None
      

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
	n,d = X.shape

	# create all of the weight matrices theta(1),...,theta(L-1)
	self.weightThetas = dict()
	
	# initialize all of the weights to be uniformly chosen from [-epsilon, epsilon]
	# note that if network has s_j units in layer j and s_j+1 units in layer j + 1,
	# theta_j has dimension s_j+1 x (s_j + 1)
	for i in range(len(self.layers) + 1):
	    # find the dimensions for the current theta matrix
	    d1 = -1
	    d2 = -1
	    # check if first theta
	    if (i == 0):
	        d2 = 2 # TODO: UPDATE THIS PLACEHOLDER
		d1 = self.layers[i]
	    
	    # check if last theta
	    if (i == len(self.layers)):
		 d2 = self.layers[i-1]
		 d1 = 1 # TODO: UPDATE THIS PLACEHOLDER

	    # update d1 and d2 if not the first or last theta
	    if (d1 == -1):
            	d1 = self.layers[i + 1]

	    if (d2 == -1):
	        d2 = self.layers[i] + 1

	    # create current theta matrix with weights uniformly chosen from [-epsilon, epsilon]
	    self.weightThetas[i+1] = (self.epsilon * 2) * np.random.random_sample((d1, d2)) - self.epsilon
	
	
	

	# unroll the weight matrices theta(1),...,theta(L-1) into a single long vector, theta,
	# that contains all parameters for the neural net
# 	theta = np.empty(len(self.layers) + 1)
# 	for i in (1, len(self.layers) + 1):
#             theta[i] = weightThetas[i]
# 
	# print theta


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
    
    
    def visualizeHiddenNodes(self, filename):
        '''
        CIS 519 ONLY - outputs a visualization of the hidden layers
        Arguments:
            filename - the filename to store the image
        '''
        
