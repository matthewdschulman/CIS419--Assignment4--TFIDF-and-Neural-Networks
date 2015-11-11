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
	self.unrolledTheta = None
	self.thetaDimensions = dict()
      

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
	n,d = X.shape

	numFeatures = X[0].size
	numClasses = np.unique(y).size
	totalLengthOfUnrolledTheta = 0

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
	        d2 = numFeatures + 1
		d1 = self.layers[i]
	    
	    # check if last theta
	    if (i == len(self.layers)):
		 d2 = self.layers[i-1] + 1
		 d1 = numClasses

	    # update d1 and d2 if not the first or last theta
	    if (d1 == -1):
            	d1 = self.layers[i + 1]

	    if (d2 == -1):
	        d2 = self.layers[i] + 1

	    # create current theta matrix with weights uniformly chosen from [-epsilon, epsilon]
	    self.weightThetas[i+1] = (self.epsilon * 2) * np.random.random_sample((d1, d2)) - self.epsilon

	    # keep track of the total length of the unrolled theta
	    totalLengthOfUnrolledTheta += (d1 * d2)
	    
	    # keep track of the individual theta dimensions
	    self.thetaDimensions[i] = [d1, d2]
	
	# unroll the weight matrices theta(1),...,theta(L-1) into a single long vector, theta,
	# that contains all parameters for the neural net
 	unrolled_theta = []
	for key in self.weightThetas:
	    cur_theta_matrix = self.weightThetas[key]
	    n,d = cur_theta_matrix.shape
	    for i in range(n):
	        for j in range(d):
		    unrolled_theta.append(cur_theta_matrix[i][j])

	# make sure unrolling worked...
	if (len(unrolled_theta) != totalLengthOfUnrolledTheta):
	    print "ERROR: UNROLLING MESSED UP!"

	self.unrolledTheta = np.array(unrolled_theta)
	forwardPropogated = self.forwardPropogation(X[0], self.unrolledTheta, self.thetaDimensions)

    def forwardPropogation(self, x, unrolledTheta, thetaDimensions):
	'''
	Takes in a vector of parameters (e.g. theta) for the neural network
	and an instance (or instances)
	and returns the neural network's outputs
	Arguments:
		theta is a dictionary of all the theta weights
		x and y equate to one labeled training instance
	Returns:
		h_theta(x_i) for any instance x_i
	'''
	cum_unroll_count = 0
	cur_a = x

	for i in range(len(self.layers) + 1):
	    # add bias unit to cur_a
	    cur_a = np.insert(cur_a, 0, 1.0)		

	    cur_theta_dimensions = thetaDimensions[i]
	    delta_unroll = cur_theta_dimensions[0] * cur_theta_dimensions[1]
            cum_unroll_count += delta_unroll
            cur_theta = np.reshape(unrolledTheta[(cum_unroll_count - delta_unroll):cum_unroll_count], (cur_theta_dimensions[0], cur_theta_dimensions[1])) 

            new_a = np.zeros(cur_theta_dimensions[0])
	    for j in range(len(new_a)):
	        # calculate a_(j+1)^(i+1)
		cum_sum = 0
	 	for k in range(cur_theta_dimensions[1]):
		    cum_sum += cur_theta[j][k]*cur_a[k]		    
		new_a[j] = self.sigmoid(cum_sum)

	    cur_a = new_a
   	
	return cur_a


    def sigmoid(self, z):
        return 1.0/(1.0 + np.exp(-z))
	

    def gradientCheck():
	'''
	Estimate gradient numerically to verify implementation
	Turn this off in final implementation of class
	'''

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
        
