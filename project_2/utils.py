import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def compute_accuracy(classifier, data_loader):
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
