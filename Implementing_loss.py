import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class layer_Dense:
    def __init__(self, n_inputs , n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs , n_neurons)
        self.biases = np.zeros((1 , n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs , self.weights) + self.biases

class Activation_ReLu:
    def forward(self, inputs):
        self.output = np.maximum(0 , inputs)

class Activation_softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs , axis=1 , keepdims=True))
        probabilities = exp_values / np.sum(exp_values , axis=1 , keepdims=True)
        self.output = probabilities
class Loss:
    def calculate(self , output , Y):
        sample_losses = self.forward(output , Y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_categoricalcrossentropy(Loss):
    def forward(self , Y_pred , Y_true):
        samples = len(Y_pred)
        Y_pred_clipped = np.clip(Y_pred , 1e-7 , 1-1e-7)

        if len(Y_true.shape) == 1 :
            correct_confidences = Y_pred_clipped[range(samples) , Y_true]

        elif len(Y_true.shape) == 2 :
            correct_confidences = np.sum(Y_pred_clipped * Y_true, axis = 1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


X , Y = spiral_data(100 , 3)

dense1 = layer_Dense(2 , 3)
activation1 = Activation_ReLu()

dense2 = layer_Dense(3 , 3)
activation2 = Activation_softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

Loss_function = Loss_categoricalcrossentropy()
loss = Loss_function.calculate(activation2.output , Y)

print("Loss:" , loss)
