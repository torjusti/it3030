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

        # Initialize the weights and biases with small random numbers
        # normalized by the square root of incoming input nodes.
        self.weights = np.random.randn(nodes, input_size) / input_size ** 0.5
        self.bias = np.random.randn(nodes, 1)

    def _linear(self, x, derivative=False):
        """ Linear activation function. """
        if derivative:
            return 1

        return x

    def _tanh(self, x, derivative=False):
        """ Hyperbolic tangent activation function. """
        if derivative:
            return 1 - np.tanh(x) ** 2

        return np.tanh(x)

    def _relu(self, x, derivative=False):
        """ ReLU activation function. """
        if derivative:
            out = np.copy(x)
            out[out <= 0] = 0
            out[out > 0] = 1
            return out

        return np.maximum(x, 0)

    def _softmax(self, x, derivative=False):
        """ Softmax activation function. """
        if derivative:
            # No need to handle this when combined with cross-entropy loss.
            # Instead, we find the delta directly without multiplying with
            # the derivative of the activation function at `z`.
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
            raise ValueError("Cross entropy must be combined with softmax at the last layer")

        self.loss_name = cost_function

        if cost_function == 'cross_entropy':
            self.loss = self._cross_entropy
        elif cost_function == 'L2':
            self.loss = self.l2_loss

        self.layers = layers

    def l2_regularization(self):
        """ Sum of squared weights across the neural network. """
        return sum(np.sum(layer.weights) for layer in self.layers)

    def l2_loss(self, output, target, regularization, derivative=False):
        """ Mean Squared Error loss function for regression. """
        if derivative:
            return output - target

        return np.sum((target - output) ** 2, axis=0) + regularization * self.l2_regularization()

    def _cross_entropy(self, output, target, regularization, derivative=False):
        """ Cross Entropy loss function for classification. """
        if derivative:
            # The cross entropy delta is handled directly in the `backpropagation` method.
            raise NotImplementedError

        return -np.sum(target * np.log(output) + (1 - target) * np.log(1 - output), axis=0) \
            + regularization * self.l2_regularization()

    def forward(self, x):
        """ Perform a forward pass through the network. """
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def update_gradient(self, error, layer, samples):
        """ Update the current minibatch gradient. Normalization is done when
        performing the actual update in `update_weights`. """
        previous_activations = samples if layer == 0 else self.layers[layer - 1].a
        self.layers[layer].weight_gradient = np.dot(error, previous_activations.transpose())
        self.layers[layer].bias_gradient = error.sum(axis=1).reshape(error.shape[0], 1)

    def update_weights(self, lr, regularization, example_count):
        """ Update all the weights in the network after computing average gradient
        from a minibatch. This also clears out the gradient in each layer. """
        for layer in self.layers:
            # Update the weights and add regularization.
            layer.weights -= lr * (layer.weight_gradient / example_count + regularization * layer.weights)
            # Update the bias vector. Regularization is not needed here, since
            # the regularization penalty function is not a function of the bias.
            layer.bias -= lr * layer.bias_gradient / example_count

    def backpropagate(self, activation, samples, targets, lr, regularization):
        """ Update the network weights using backpropagation for gradient computation. """
        last_layer = self.layers[len(self.layers) - 1]

        # Compute the delta for the last layer.
        if self.loss_name == 'cross_entropy':
            # If we assume that softmax is used for the last layer, this is the backprop delta.
            error = last_layer.a - targets
        else:
            # Standard backpropagation base case.
            error = self.loss(activation, targets, regularization, derivative=True) \
                * last_layer.activation( last_layer.z, derivative=True)

        self.update_gradient(error, len(self.layers) - 1, samples)

        for i in range(len(self.layers) - 2, -1, -1):
            # Compute the error for the next-up layer.
            error = self.layers[i].activation(self.layers[i].z, derivative=True) \
                * np.dot(self.layers[i + 1].weights.transpose(), error)

            self.update_gradient(error, i, samples)

    def train(self, data, labels, val_data=None, val_labels=None, epochs=50,
              lr=0.01, batch_size=8, regularization=0):
        """ Train the network. """
        train_losses = []
        val_losses = []

        for epoch in range(1, epochs + 1):
            train_loss = 0

            for i in range(0, data.shape[1], batch_size):
                # Get training data for this minipatch.
                samples = data[:, i:i+batch_size]
                # Get labels corresponding to this minibatch.
                targets = labels[:, i:i+batch_size]
                # Pass the entire minibatch through the network at once.
                activation = self.forward(samples)
                # Add the training loss for this minibatch.
                train_loss += self.loss(activation, targets, regularization).sum()
                # Compute the network gradients using backpropagation.
                self.backpropagate(activation, samples, targets, lr, regularization)
                # Update the weights using gradient descent.
                self.update_weights(lr, regularization, samples.shape[1])

            train_loss /= data.shape[1]
            train_losses.append(train_loss)

            validate = val_data is not None and val_labels is not None

            if validate:
                # Compute loss on validation set.
                val_loss = self.loss(self.forward(val_data), val_labels, regularization).sum()
                val_loss /= val_data.shape[1]
                val_losses.append(val_loss)

                # Compute accuracy on validation set.
                predictions = np.argmax(self.forward(val_data), axis=0)
                targets = np.argmax(val_labels, axis=0)
                correct = np.count_nonzero(predictions == targets)
                accuracy = correct / val_data.shape[1]


            # Animate addition of new losses.
            plt.axis(ylim=[0, max(train_losses + val_losses)])
            plt.plot(train_losses, 'b')
            plt.plot(val_losses, 'r')
            plt.pause(0.05)

            if validate:
                print(f'Epoch {epoch}: training loss {train_loss}, validation loss {val_loss}, accuracy {accuracy}')
            else:
                print(f'Epoch {epoch}: training loss {train_loss}')

        plt.show()

    def dump_weights(self, filename):
        """ Dump learned weights to file as a string. """
        with open(filename, 'w') as f:
            # Disable truncation in output.
            with np.printoptions(threshold=np.inf):
                for layer in self.layers:
                    f.write(str(layer.weights))
                    f.write(str(layer.bias))
                    f.write('\n')


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
        val_labels = one_hot(val_labels)

    # Read layer sizes from config file.
    layer_sizes = [int(size.strip()) for size in config['MODEL']['layers'].split(',')]
    # Read layer activation functions from config file.
    layer_activations = [a.strip() for a in config['MODEL']['activations'].split(',')]

    layers = []

    previous_width = train_data.shape[0]

    # Add hidden layers.
    for i, layer in enumerate(layer_sizes):
        layers.append(FullyConnectedLayer(previous_width, layer,
            activation=layer_activations[i]))

        previous_width = layer

    # Add output layer.
    if config['MODEL']['loss_type'] == 'cross_entropy':
        layers.append(FullyConnectedLayer(previous_width,
            val_labels.shape[0], activation='softmax'))
    elif config['MODEL']['loss_type'] == 'L2':
        layers.append(FullyConnectedLayer(previous_width, 1, activation='linear'))

    network = SequentialNetwork(layers, cost_function=config['MODEL']['loss_type'])

    network.train(train_data, train_labels, val_data, val_labels,
        lr=float(config['HYPER']['learning_rate']), epochs=int(config['HYPER']['no_epochs']),
        regularization=float(config['HYPER']['L2_regularization']), batch_size=8)

    network.dump_weights('weights.txt')


if __name__ == '__main__':
    main()
