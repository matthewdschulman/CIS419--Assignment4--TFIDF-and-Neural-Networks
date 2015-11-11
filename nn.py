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
	self.numFeatures = -1
	self.numClasses = -1
	self.numTrainingInstances = -1
	self.maxNodesInALayer = -1
      

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
	n,d = X.shape
	self.numTrainingInstances = n

	self.numFeatures = X[0].size
	self.numClasses = np.unique(y).size
	self.numLayers = len(self.layers) + 2
	self.maxNodesInALayer = max(max(self.layers),self.numFeatures)
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
	        d2 = self.numFeatures + 1
		d1 = self.layers[i]
	    
	    # check if last theta
	    if (i == len(self.layers)):
		 d2 = self.layers[i-1] + 1
		 d1 = self.numClasses

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
	
	# Backpropogation to minimize J(theta)

	for e in range(1): #TODO:Should be number of epochs
            print "======= NEW EPOCH ======="
       	    n,d = X.shape
            l = self.numLayers
            I = self.numTrainingInstances
            j = self.maxNodesInALayer
            # Set delta_ij^(i) = 0 for all l,i,j
            # grad_l = [[[0 for k in xrange(j)] for j in xrange(l)] for i in xrange(I)]
	    grad_l = []

	    for i in range(2):#TODO:Should be I
        	a_1 = X[i]

        	# Compute {a^2,...a^L} via forward propogation
        	all_a_Ls = self.forwardPropogation(a_1, self.unrolledTheta, self.thetaDimensions)

        	# Compute delta_l = a_l - y_i 
        	delta_L = all_a_Ls[-1] - y[i]

        	# Compute errors {delta^(L-1),...delta^(2)}
        	errors = dict()
        	for x in range(len(all_a_Ls)):
                    errors[x] = all_a_Ls[x] - y[i]

	
        	# Compuyte gradients delta_ij^(l) = delta_ij^(l) + a_j^(l)delta_i^(l + 1)
		cur_i_grad_l = []
         	for x in range(l-2):
		    cur_j_grad_l = []
		    for z in range(len(all_a_Ls[x+1])):
			cur_a_val = all_a_Ls[x][z]
			cur_error_val = errors[x+1][z]
			cur_j_grad_l.append(cur_a_val*cur_error_val)
		    cur_i_grad_l.append(cur_j_grad_l)

		grad_l.append(cur_i_grad_l)

            # Compute avg regularized gradient D_ij^(l)
            # D_ijl = (1.0/n)*grad_l[]
	    d_matrix = dict()
	    for i in range(2):#TODO: SHOULD BE I
		cur_l = []
 	        for x in range(l-2):
		    cur_j = []
		    for z in range(len(all_a_Ls[x+1])):
			cur_val_to_append = (1.0/n) * grad_l[i][x][z]
			if z != 0:
			    cur_val_to_append += self.learningRate*1 #TODO: UPDATE WITH PROPER THETA VALUE  
		        cur_j.append(cur_val_to_append)
		    cur_l.append(cur_j)
		d_matrix[i] = cur_l


	    # Update weights via gradient step
	    # for sampleNumb in range(2): #TODO SHOULD BE I
	    #     cum_unroll_count = 0
	    #     for i in range(len(self.layers) + 1):

	    #         cur_theta_dimensions = self.thetaDimensions[i]
	    #         delta_unroll = cur_theta_dimensions[0] * cur_theta_dimensions[1]
            #         cum_unroll_count += delta_unroll
            #         cur_theta = np.reshape(self.unrolledTheta[(cum_unroll_count - delta_unroll):cum_unroll_count], (cur_theta_dimensions[0], cur_theta_dimensions[1])) 
	    #         cur_theta = cur_theta[sampleNumb]
	    #         cur_d_matrix = d_matrix[sampleNumb]
	    #         print "cur_theta"
	    #         print cur_theta
	    #         print "cur_d"
	    #         print cur_d_matrix

	    # 	    for q in range(cur_theta_dimensions[0]):
	    # 	        cur_theta[i][q] = cur_theta[i][q] - cur_d_matrix[i][q] 

    def forwardPropogation(self, x, unrolledTheta, thetaDimensions):
	'''
	Takes in a vector of parameters (e.g. theta) for the neural network
	and an instance
	and returns the neural network's outputs
	Arguments:
		theta is a dictionary of all the theta weights
		x and y equate to one labeled training instance
	Returns:
		list of all of the a_l's...
		...last element of list is h_theta(x_i) for any instance x_i
	'''
	all_a_ls = []

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
	    all_a_ls.append(cur_a)
   	
	return all_a_ls


    def sigmoid(self, z):
        return 1.0/(1.0 + np.exp(-z))
	

#     def computeCost(self, X, unrolledTheta, thetaDimensions, y):
# 	'''
# 	Compute the cost, J(theta)
# 	'''
#         n,d = X.shape
# 	print n
# 	print d
# 	
#         cost1 = 0
#         for i in range(n):
# 	    h_theta = self.forwardPropogation(X[i], self.unrolledTheta, self.thetaDimensions)
#             for k in range(self.numClasses):
# 		true = y[i]
#  		predicted = h_theta[k]
#                 cost1 += (true*np.log(predicted) + (1-true)*np.log(1-predicted)) 
# 	cost2 = 0
# #
#  

    def gradientCheck():
	'''
	Estimate gradient numerically to verify implementation
	Turn this off in final implementation of class
	Compare the partial derivative of J(theta) computed using
	backpropogation vs. the numerical gradient estimate and
	make sure they're in line with each other
	'''

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
	# forward propogate with all x_i's in X
	# then return the last a vector's predictions...
    
    def visualizeHiddenNodes(self, filename):
        '''
        CIS 519 ONLY - outputs a visualization of the hidden layers
        Arguments:
            filename - the filename to store the image
        '''
        
