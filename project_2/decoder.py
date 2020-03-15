import torch
import torch.nn.functional as F
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
        )

    def forward(self, x):
        size = (self.input_dim[1], self.input_dim[2])
        x = self.linear_layers(x)
        x = x.view((x.shape[0], 32, 24, 24))
        return torch.tanh(F.interpolate(self.conv_layers(x), size, mode='bilinear',
                                        align_corners=False))
