import numpy as np

#input layer with 4 neurons
inputs  = [1 , 2 , 3 , 2.5]

#distinct weight for each neuron
weights = [[0.2 , 0.8 , -0.5 , 1],
           [0.5 , -0.91 , 0.26 , -0.5],
           [-0.26 , -0.27 , 0.17 , 0.87]]

#distinct bias
biases    = [2 , 3 , 0.5]

output = np.dot(weights , inputs) + biases
print(output)
