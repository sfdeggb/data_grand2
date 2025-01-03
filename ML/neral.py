import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
	features = np.array(features)
	labels= np.array(labels)
	weights = np.array(weights)
	
	outputs  = features.dot(weights.T)+bias
	prob = 1 / (1 + np.exp(-outputs))
	mse = sum((labels - prob)**2)/ len(labels)
	return prob, mse

if __name__ == "__main__":
    features = [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]]
    labels = [0, 1, 0]
    weights = [0.7, -0.4]
    bias = -0.1
    prob, mse = single_neuron_model(features, labels, weights, bias)
    print(prob, mse)