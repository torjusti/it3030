import torch
import numpy as np


class Classifier(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(np.prod(input_dim), 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)
