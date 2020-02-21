import torch
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def compute_accuracy(classifier, data_loader, device=device):
    """ Utility function to compute the accuracy of a classifier on a dataset. """
    with torch.no_grad():
        num_correct = 0
        total = 0

        for images, labels in data_loader:
            activations = classifier(images.to(device)).cpu()
            predictions = torch.max(activations, 1)
            num_correct += (predictions.indices == labels).sum().item()
            total += labels.shape[0]

        return num_correct / total


def flatten_images(images, rows, cols):
    """ Merge an array of images into one big image. """
    width, height = images.shape[1], images.shape[2]
    image = np.zeros(shape=(rows * height, cols * width))

    i = 0

    for row in range(rows):
        for col in range(cols):
            image[row * height : (row + 1) * height, col * width : (col + 1) * width] = images[i]
            i += 1

    return image
