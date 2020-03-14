import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def compute_accuracy(classifier, data_loader, device=device):
    classifier.eval()

    """ Utility function to compute the accuracy of a classifier on a dataset. """
    with torch.no_grad():
        num_correct = 0
        total = 0

        for images, labels in data_loader:
            activations = classifier(images.to(device)).cpu()
            predictions = torch.max(activations, 1)
            num_correct += (predictions.indices == labels).sum().item()
            total += labels.shape[0]

    classifier.train()

    return num_correct / total


def flatten_images(images, rows, cols):
    """ Merge an array of images into one big image. """
    if len(images.shape) == 4:
        depth, width, height = images.shape[1:]
    else:
        width, height = images.shape[1:]
        depth = None

    if depth:
        image = np.zeros(shape=(rows * height, cols * width, depth))
    else:
        image = np.zeros(shape=(rows * height, cols * width))

    i = 0

    for row in range(rows):
        for col in range(cols):
            if i >= len(images):
                break

            if depth:
                image[row * height : (row + 1) * height, col * width : (col + 1) * width, :] = \
                    np.rollaxis(images[i], 0, 3)
            else:
                image[row * height : (row + 1) * height, col * width : (col + 1) * width] = images[i]

            i += 1

    return image


def tsne_plot(encoder, loader, title):
    """ Visualize autoencoder encodings. """
    encoder.eval()

    with torch.no_grad():
        # Gather embeddings for the D1 validation data.
        embedding_classes = []
        embeddings = []

        for images, targets in loader:
            for i in range(len(images)):
                embeddings.append(encoder(torch.unsqueeze(
                    images[i], 0).to(device)).cpu().squeeze().numpy())

                embedding_classes.append(targets[i])

                if len(embeddings) == 256:
                    break

            if len(embeddings) == 256:
                break

        transformed_coordinates = TSNE(n_components=2).fit_transform(np.array(embeddings))
        plt.scatter(transformed_coordinates[:, 0], transformed_coordinates[:, 1], c=embedding_classes)
        plt.title(title)
        plt.show()

    encoder.train()
