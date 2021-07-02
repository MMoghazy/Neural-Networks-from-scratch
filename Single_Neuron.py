#input layer with 3 neurons
inputs  = [1.2 , 5.1 ,  2.1]

#distinct weight for each neuron
weights = [3.1 , 2.1 , 8.7]

#distinct bias
bias    = 3

# output = input*weight + bias
output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
print(output)
