import torch
import numpy as np


class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, input_dim):
        super().__init__()

        self.input_dim = input_dim

        self.network = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, np.prod(input_dim)),
        )

    def forward(self, x):
        x = self.network(x)
        return x.view((x.shape[0], *self.input_dim))
