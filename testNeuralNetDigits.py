"""
======================================================
Test the neural network model
======================================================

Author: Matt Schulman, 2015

"""
print(__doc__)

import numpy as np
from numpy import loadtxt
from sklearn import datasets
from sklearn.metrics import accuracy_score

from nn import NeuralNet

# learning rate parameters to be trained by hand
numEpochs = 100
learningRate = 0.3
epsilon = 0.12
regularization_parameter = 0.001
num_of_hidden_layers = 1
nodes_in_hidden_layer = 100
layers = []

# load the data
filenameX = 'data/digitsX.dat'
dataX = loadtxt(filenameX, delimiter=',')
filenameY = 'data/digitsY.dat'
dataY = loadtxt(filenameY, delimiter=',')
n,d = dataX.shape

# create NeuralNet class
modelNN = NeuralNet(layers, epsilon, learningRate, numEpochs)

# train neural network on digits data

# find the training accuracy

# report the training accuracy
