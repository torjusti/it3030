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

        # Initialize the weights and biases with small random numbers.
        # TODO: Should this be scaled by some factor? ReLU is dying.
        self.weights = np.random.randn(nodes, input_size)
        self.bias = np.random.randn(nodes, 1)

        # Gradients which are computed using backpropagation are initially
        # set to 0 to simplify computations later on in `SequentialNetwork`.
        self.weight_gradient = np.zeros(self.weights.shape)
        self.bias_gradient = np.zeros(self.bias.shape)

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

    def l2_loss(self, output, target, derivative=False):
        """ Mean Squared Error loss function for regression. """
        if derivative:
            # TODO: Do we need the factor of 2 outside this?
            return output - target

        # TODO: Should regularization be added here, or can it be ignored?
        return np.sum((target - output) ** 2)

    def _cross_entropy(self, output, target, derivative=False):
        """ Cross Entropy loss function for classification. """
        # TODO: Should regularization be added here, or can it be ignored?
        # TODO: This might not be completely equal to the slide formula.
        return -np.sum(target * np.log(output))

    def forward(self, x):
        """ Perform a forward pass through the network. """
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def update_gradient(self, error, layer, input_vector):
        """ Update the current minibatch gradient. Normalization is done when
        performing the actual update in `update_weights`. """
        previous_activations = input_vector if layer == 0 else self.layers[layer - 1].a
        self.layers[layer].weight_gradient += np.dot(error, previous_activations.transpose())
        self.layers[layer].bias_gradient += error

    def update_weights(self, lr, regularization, example_count):
        """ Update all the weights in the network after computing average gradient\
        from a minibatch. This also clears out the gradient in each layer. """
        for layer in self.layers:
            # Update the weights and add regularization.
            # TODO: Should regularization be scaled with number of training examples?
            # TODO: What is the difference between alpha and lambda guys?
            # TODO: In slides, regularization has a constant factor. Do we need it?
            layer.weights -= lr * (layer.weight_gradient / example_count + regularization * layer.weights)
            # Update the bias vector. Regularization is not needed here, since
            # the regularization penalty function is not a function of the bias.
            # TODO: Should this also have a regularization term, since the assignment mentions that?
            layer.bias -= lr * layer.bias_gradient / example_count
            # Clear out the gradient for the next minibatch.
            layer.weight_gradient[:] = 0
            layer.bias_gradient[:] = 0

    def backpropagate(self, activation, input_vector, target, lr, regularization):
        """ Update the network weights using backpropagation for gradient computation. """
        last_layer = self.layers[len(self.layers) - 1]

        # Compute the delta for the last layer.
        if self.loss_name == 'cross_entropy':
            # If we assume that softmax is used for the
            # last layer, this is the backprop delta.
            error = last_layer.a - target
        else:
            # Standard backpropagation base case.
            error = self.loss(activation, target, derivative=True) * last_layer.activation(
                last_layer.z, derivative=True)

        self.update_gradient(error, len(self.layers) - 1, input_vector)

        for i in range(len(self.layers) - 2, -1, -1):
            # Compute the error for the next-up layer.
            error = self.layers[i].activation(self.layers[i].z, derivative=True) \
                * np.dot(self.layers[i + 1].weights.transpose(), error)

            self.update_gradient(error, i, input_vector)

    def predict(self, x):
        """ Perform a prediction using the network. """
        activation = self.forward(x)

        # Handle classification with argmax.
        if self.loss_name == 'cross_entropy':
            return np.argmax(activation)

        # Get number from matrix.
        return np.squeeze(x)

    def train(self, data, labels, val_data, val_labels, epochs=50,
              lr=0.01, batch_size=8, regularization=0):
        """ Train the network. """
        train_losses = []
        val_losses = []

        for epoch in range(1, epochs + 1):
            train_loss = 0

            for i in range(data.shape[1]):
                # We lose a dimension when indexing, so reshape data to a column vector.
                target = labels[:, i].reshape(labels.shape[0], 1)
                input_vector = data[:, i].reshape(data.shape[0], 1)

                activation = self.forward(input_vector)

                train_loss += self.loss(activation, target)

                # Update gradient using backpropagation.
                self.backpropagate(activation, input_vector, target, lr, regularization)

                # Update gradient at the end of each batch, or at last sample.
                if i % batch_size == (batch_size - 1) or i == data.shape[1] - 1:
                    # Handle non-complete batches at end of dataset.
                    effective_batch_size = (i % batch_size) + 1
                    # Update weights with computed gradient.
                    self.update_weights(lr, regularization, effective_batch_size)

            val_loss = 0

            for i in range(val_data.shape[1]):
                # Note that for classification, the validation data is not one-hot encoded.
                target = val_labels[:, i].reshape(val_labels.shape[0], 1)
                input_vector = val_data[:, i].reshape(val_data.shape[0], 1)

                activation = self.forward(input_vector)

                val_loss += self.loss(activation, target)

            train_loss /= data.shape[1]
            val_loss /= val_data.shape[1]

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Animate addition of new losses.
            plt.axis(ylim=[0, max(train_losses + val_losses)])
            plt.plot(train_losses, 'b')
            plt.plot(val_losses, 'r')
            plt.pause(0.05)

            print(f'Epoch {epoch}: training loss {train_loss}, validation loss {val_loss}')

        plt.show()

    def dump_weights(self, filename):
        """ Dump learned weights to file as a string. """
        with open(filename, 'w') as f:
            # Disable truncation in output.
            with np.printoptions(threshold=np.inf):
                for layer in self.layers:
                    # TODO: Skal vi serr ikke bruke NumPy greiene for å lagre til fil?
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

    # Read layer sizes from config file.
    layer_sizes = [int(size.strip()) for size in config['MODEL']['layers'].split(',')]
    # Read layer activation functions from config file.
    layer_activations = [a.strip() for a in config['MODEL']['activations'].split(',')]

    layers = []

    previous_width = train_data.shape[0]

    # TODO: Handle the hidden layer shit correctly, and the no hidden layer shit.
    # TODO: Regression without hidden layer should be fine. But should classification have multiple logits?
    for i, layer in enumerate(layer_sizes):
        layers.append(FullyConnectedLayer(previous_width, layer,
            activation=layer_activations[i]))

        previous_width = layer

    network = SequentialNetwork(layers, cost_function=config['MODEL']['loss_type'])

    network.train(train_data, train_labels, val_data, val_labels,
        lr=float(config['HYPER']['learning_rate']), epochs=int(config['HYPER']['no_epochs']),
        regularization=float(config['HYPER']['L2_regularization']), batch_size=8)

    # TODO: When to dump? After all epochs finished? What format?
    network.dump_weights('weights.txt')


if __name__ == '__main__':
    main()
