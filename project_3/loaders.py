import numpy as np
import torch


class ArrayDataset(torch.utils.data.Dataset):
    def __init__(self, images, targets):
        """ Dataset for NumPy data. """
        self.images = images
        self.targets = targets

    def __getitem__(self, index):
        return self.images[index], self.targets[index]

    def __len__(self):
        return self.images.shape[0]


def get_data_loader(data_set, training=True):
    """ Returns a PyTorch data loader for the given TensorFlow dataset. """
    images, labels = data_set.get_full_data_set(training=training)
    data_set = ArrayDataset(np.moveaxis(images, 3, 1), labels)
    return torch.utils.data.DataLoader(data_set,
        shuffle=True, batch_size=256)
