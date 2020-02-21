import torch


class LatentClassifier(torch.nn.Module):
    def __init__(self, encoder, latent_dim, num_classes, freeze=False):
        super().__init__()

        self.encoder = encoder

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, num_classes),
        )

        if freeze:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = True

    def forward(self, x):
        return self.classifier(self.encoder(x))
