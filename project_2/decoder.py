import torch


class Decoder(torch.nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

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
            torch.nn.ConvTranspose2d(16, 1, 3),
        )

    def forward(self, x):
        x = self.linear_layers(x)
        x = x.view((x.shape[0], 32, 24, 24))
        return self.conv_layers(x)
