import torch
from encoder import Encoder
from decoder import Decoder


class AutoEncoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(input_dim, latent_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))
