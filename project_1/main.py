import numpy as np
from configparser import ConfigParser
from scipy.special import softmax
import matplotlib.pyplot as plt
import sys

np.random.seed(42)


class FullyConnectedLayer:
    def __init__(self, input_size, nodes, activation='relu'):
        """ Initialize a new fully-connected layer, given the amount of nodes
        on the previous layer, `input_size`, the number of nodes at this layer
        as determined by `nodes`, and an activation function `activation`. """
        self.activation_name = activation

        # Store a pointer to the selected activation function.
        if activation == 'linear':
            self.activation = self._linear
        elif activation == 'relu':
            self.activation = self._relu
        elif activation == 'tanh':
            self.activation = self._tanh
        elif activation == 'softmax':
            self.activation = self._softmax
        elif activation == 'sigmoid':
            self.activation = self._sigmoid

        # Initialize the weights and biases with small random numbers.
        self.weights = np.random.randn(nodes, input_size)
        self.bias = np.random.randn(nodes, 1)

    def _linear(self, x, derivative=False):
        """ Linear activation function. """
        if derivative:
            return 1

        return x

    def _tanh(self, x, derivative=False):
        """ Hyperbolic tangent activation function. """
        if derivative:
            return (np.cosh(x) ** 2 - np.sinh(x) ** 2) / np.cosh(x) ** 2

        return np.tanh(x)

    def _relu(self, x, derivative=False):
        """ ReLU activation function. """
        if derivative:
            return np.sign(x)

        return np.maximum(x, 0)

    def _softmax(self, x, derivative=False):
        """ Softmax activation function. """
        if derivative:
            # No need to handle this when combined with cross-entropy loss.
            raise NotImplementedError

        return softmax(x, axis=0)

    def _sigmoid(self, x, derivative=False):
        """ Sigmoid activation function. """
        if derivative:
            return self._sigmoid(x) * (1 - self._sigmoid(x))

        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        # Perform the linear combination of weights and inputs, and add the bias.
        self.z = np.dot(self.weights, x) + self.bias

        # Perform the activation function on the combined weights and inputs.
        # The result is cached in the layer object for use in backpropagation.
        self.a = self.activation(self.z)

        return self.a


class SequentialNetwork:
    def __init__(self, layers, cost_function='cross_entropy'):
        """ Initialize a neural network by defining its layers. """
        if cost_function == 'cross_entropy' and layers[-1].activation_name != 'softmax':
            raise ValueError(
                "Cross entropy must be combined with softmax at the last layer")

        self.loss_name = cost_function

        # Store a pointer to the cost function.
        if cost_function == 'cross_entropy':
            self.loss = self._cross_entropy
        elif cost_function == 'L2':
            self.loss = self.l2_loss

        self.layers = layers

    def l2_loss(self, output, target, derivative=False):
        """ Mean Squared Error loss function for regression. """
        if derivative:
            # TODO: Check for correctness.
            return output - target

        # TODO: Not used yet.
        # TODO: Should regularization be added here? Probably.
        return np.sum((output - target) ** 2)

    def _cross_entropy(self, output, target, derivative=False):
        """ Cross Entropy loss function for classification. """
        # TODO: Not used yet.
        # TODO: Should regularization be added here? Probably.
        return -np.dot(target, np.log(output))

    def forward(self, x):
        """ Perform a forward pass through the network. """
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def update_weights(self, error, layer, input_vector, lr, regularization):
        """ Update the network weights using gradient descent. """
        previous_activations = input_vector if layer == 0 else self.layers[layer - 1].a

        # TODO: Should regularization be scaled with number of training examples?
        self.layers[layer].weights -= lr * (np.dot(error, previous_activations.transpose()) \
            + regularization * self.layers[layer].weights)

        # TODO: Should the bias also have a regularization term?
        self.layers[layer].bias -= self.layers[layer].bias - lr * error

    def backpropagate(self, activation, input_vector, target, lr, regularization):
        """ Update the network weights using backpropagation for gradient computation. """
        # Grab a pointer to the last layer.
        last_layer = self.layers[len(self.layers) - 1]

        # Compute the delta for the last layer.
        if self.loss_name == 'cross_entropy':
            # If we assume that softmax is used for the last layer.
            error = last_layer.a - target
        else:
            # Standard backpropagation base case.
            error = self.loss(activation, target, derivative=True) * last_layer.activation(
                last_layer.z, derivative=True)

        self.update_weights(error, len(self.layers) - 1, input_vector, lr, regularization)

        for i in range(len(self.layers) - 2, -1, -1):
            # Compute the error for the next-up layer.
            error = self.layers[i].activation(self.layers[i].z, derivative=True) \
                * np.dot(self.layers[i + 1].weights.transpose(), error)

            self.update_weights(error, i, input_vector, lr, regularization)

    def train(self, data, labels, epochs=50, lr=0.01, regularization=0):
        """ Train the network. """
        for epoch in range(1, epochs + 1):
            # Count number of correct predictions for computing accuracy.
            correct = 0

            for i in range(data.shape[1]):
                # Ensure input and target data have both dimensions set.
                target = labels[:, i].reshape(labels.shape[0], 1)
                input_vector = data[:, i].reshape(data.shape[0], 1)

                # Perform a forward pass.
                activation = self.forward(input_vector)

                # Check if the prediction was correct.
                if np.argmax(activation) == np.argmax(target):
                    correct += 1

                self.backpropagate(activation, input_vector, target, lr, regularization)

            # Compute accuracy for epoch.
            accuracy = correct / data.shape[1]

            print(f'Epoch {epoch}: training accuracy {accuracy}')


def load_data(filename):
    """ Loads a CSV dataset from the given file. """
    with open(filename) as training_file:
        # Split the lines on commas and convert data to floats.
        data = np.array([list(map(float, line.split(','))) for line in training_file.readlines()])
        # Extract label from dataset and return.
        return np.transpose(data[:, :-1]), np.array([data[:, -1]])


def one_hot(labels):
    """ Convert labels to one-hot encoded labels. """
    one_hot = np.zeros((labels.size, int(labels.max()+1)))
    one_hot[np.arange(labels.size), labels.astype('int')] = 1
    return one_hot.transpose()


def main():
    """ Read the configuration file and train the network. """
    if len(sys.argv) < 2:
        print('No configuration file provided')

    config = ConfigParser()
    config.read(sys.argv[1])

    train_data, train_labels = load_data(config['DATA']['training'])
    val_data, val_labels = load_data(config['DATA']['validation'])

    # If loss is cross-entropy, we assume a classification problem.
    if config['MODEL']['loss_type'] == 'cross_entropy':
        train_labels = one_hot(train_labels)

    # Read layer sizes from config file.
    layer_sizes = [int(size.strip()) for size in config['MODEL']['layers'].split(',')]
    # Read layer activation functions from config file.
    layer_activations = [a.strip() for a in config['MODEL']['activations'].split(',')]

    layers = []

    previous_width = train_data.shape[0]

    # TODO: Handle the hidden layer shit correctly, and the no hidden layer shit.
    for i, layer in enumerate(layer_sizes):
        layers.append(FullyConnectedLayer(previous_width, layer,
            activation=layer_activations[i]))

        previous_width = layer

    network = SequentialNetwork(layers, cost_function=config['MODEL']['loss_type'])

    network.train(train_data, train_labels,
        lr=float(config['HYPER']['learning_rate']), epochs=int(config['HYPER']['no_epochs']),
        regularization=float(config['HYPER']['L2_regularization']))

    # TODO: Validate trained model.
    # TODO: Dump training data to file.

if __name__ == '__main__':
    main()
