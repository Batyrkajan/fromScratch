import numpy as np

# Define sigmoid activation function and its derivative
def sigmoid(x):
    return (1/(1+ np.exp(-x)))

def sigmoid_derivative(x):
    return x* (1-x)

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Input dataset (XOR inputs)
inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Target outputs
outputs = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Initialize weights and biases randomly
np.random.seed(42)  # For reproducibility
input_layer_neurons = 2  # Number of input features
hidden_layer_neurons = 2  # Number of hidden layer neurons
output_neurons = 1  # Number of output neurons

# Weights and biases
weights_input_hidden = np.random.uniform(size=(input_layer_neurons,hidden_layer_neurons))
bias_hidden = np.random.uniform(size=(1, hidden_layer_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))

# Learning rate
learning_rate = 0.1

# Training the network
for epoch in range(10000):  # 10,000 iterations
    # Forward propagation
    hidden_layer_input = np.dot(inputs, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Compute loss (for monitoring)
    loss = mse_loss(outputs, predicted_output)

    # Calculate the error
    error = outputs - predicted_output

    # Backpropagation
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden += inputs.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

# Output results
print("Final predicted outputs:")
print(predicted_output.round(2))