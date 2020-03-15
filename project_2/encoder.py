import torch


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim[0], 16, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3),
            torch.nn.AdaptiveMaxPool2d((24, 24)),
            torch.nn.ReLU(),

            torch.nn.Flatten(),
            torch.nn.Linear(24 * 24 * 32, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, latent_dim),
        )

    def forward(self, x):
        return self.network(x)
