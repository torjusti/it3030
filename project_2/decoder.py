import torch
import numpy as np


class Decoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.input_dim = input_dim

        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 24 * 24 * 32),
        )

        self.conv_layers = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(32, 16, 3),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, input_dim[0], 3),
            torch.nn.Flatten(),
            torch.nn.Linear(input_dim[0] * 28 * 28, np.prod(input_dim)),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        x = self.linear_layers(x)
        x = x.view((x.shape[0], 32, 24, 24))
        return self.conv_layers(x).reshape(x.shape[0], *self.input_dim)
