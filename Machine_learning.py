import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

training_output = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3, 1))-1


for iteration in range(1000):

    input_layer = training_inputs
    output = sigmoid(np.dot(input_layer, synaptic_weights))
    error = training_output - output
    derivative = sigmoid_derivative(output)
    adjustment = error * derivative
    synaptic_weights += np.dot(input_layer.T, adjustment)



print ("input layer")
print (input_layer)
print ("Error")
print (error)
print ("sigmoid derivate")
print (derivative)
print ("adjustement")
print (adjustment)
print ("synaptic weights")
print (synaptic_weights)
print ("This is the output")
print (output)

