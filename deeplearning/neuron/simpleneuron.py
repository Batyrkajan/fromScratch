import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

# Define sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Generate non-linear dataset (circle-in-circle)
inputs, outputs = make_circles(n_samples=500, noise=0.05, factor=0.5)
outputs = outputs.reshape(-1, 1)  # Reshape for compatibility

# Split into training and validation data
train_inputs, val_inputs, train_outputs, val_outputs = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

# Initialize weights and biases randomly
np.random.seed(42)  # For reproducibility
input_layer_neurons = 2  # Number of input features
hidden_layer_neurons = 2  # Number of neurons in the first hidden layer
hidden_layer2_neurons = 4  # Number of neurons in the second hidden layer
output_neurons = 1  # Number of output neurons

weights_input_hidden = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
bias_hidden = np.random.uniform(size=(1, hidden_layer_neurons))
weights_hidden1_hidden2 = np.random.uniform(size=(hidden_layer_neurons, hidden_layer2_neurons))
bias_hidden2 = np.random.uniform(size=(1, hidden_layer2_neurons))
weights_hidden2_output = np.random.uniform(size=(hidden_layer2_neurons, output_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))

# Regularization parameters
l2_lambda = 0.01  # Regularization strength
dropout_rate = 0.5  # Dropout rate (50%)

# Learning rate
learning_rate = 0.1

# Training the network
previous_validation_loss = float('inf')
for epoch in range(10000):
    # Forward propagation
    hidden_layer_input = np.dot(train_inputs, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    # Apply dropout to the first hidden layer
    dropout_mask1 = (np.random.rand(*hidden_layer_output.shape) > dropout_rate).astype(float)
    hidden_layer_output *= dropout_mask1
    hidden_layer_output /= (1 - dropout_rate)

    hidden_layer2_input = np.dot(hidden_layer_output, weights_hidden1_hidden2) + bias_hidden2
    hidden_layer2_output = sigmoid(hidden_layer2_input)

    # Apply dropout to the second hidden layer
    dropout_mask2 = (np.random.rand(*hidden_layer2_output.shape) > dropout_rate).astype(float)
    hidden_layer2_output *= dropout_mask2
    hidden_layer2_output /= (1 - dropout_rate)

    output_layer_input = np.dot(hidden_layer2_output, weights_hidden2_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Compute loss with L2 regularization
    loss = mse_loss(train_outputs, predicted_output)
    l2_penalty = l2_lambda * (
        np.sum(weights_input_hidden ** 2) +
        np.sum(weights_hidden1_hidden2 ** 2) +
        np.sum(weights_hidden2_output ** 2)
    )
    loss += l2_penalty

    # Backpropagation
    d_output = (train_outputs - predicted_output) * sigmoid_derivative(predicted_output)

    error_hidden_layer2 = d_output.dot(weights_hidden2_output.T)
    d_hidden_layer2 = error_hidden_layer2 * sigmoid_derivative(hidden_layer2_output)

    error_hidden_layer = d_hidden_layer2.dot(weights_hidden1_hidden2.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases with L2 regularization
    weights_hidden2_output += learning_rate * (hidden_layer2_output.T.dot(d_output) - l2_lambda * weights_hidden2_output)
    bias_output += learning_rate * np.sum(d_output, axis=0, keepdims=True)

    weights_hidden1_hidden2 += learning_rate * (hidden_layer_output.T.dot(d_hidden_layer2) - l2_lambda * weights_hidden1_hidden2)
    bias_hidden2 += learning_rate * np.sum(d_hidden_layer2, axis=0, keepdims=True)

    weights_input_hidden += learning_rate * (train_inputs.T.dot(d_hidden_layer) - l2_lambda * weights_input_hidden)
    bias_hidden += learning_rate * np.sum(d_hidden_layer, axis=0, keepdims=True)

    # Validation loss for early stopping
    val_hidden_layer = sigmoid(np.dot(val_inputs, weights_input_hidden) + bias_hidden)
    val_hidden_layer2 = sigmoid(np.dot(val_hidden_layer, weights_hidden1_hidden2) + bias_hidden2)
    predicted_validation = sigmoid(np.dot(val_hidden_layer2, weights_hidden2_output) + bias_output)
    validation_loss = mse_loss(val_outputs, predicted_validation)

    if epoch > 10 and validation_loss > previous_validation_loss:
        print(f"Early stopping triggered at epoch {epoch}!")
        break
    previous_validation_loss = validation_loss

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Training Loss: {loss:.4f}, Validation Loss: {validation_loss:.4f}")

# Output results
print("Final predicted outputs (training set):")
print(predicted_output.round(2))
