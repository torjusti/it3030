import torch


class ClassifierHead(torch.nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class LatentClassifier(torch.nn.Module):
    def __init__(self, encoder, latent_dim, num_classes, freeze=False):
        super().__init__()

        self.encoder = encoder

        # Add classifier head which classifies using embedding.
        self.classifier = ClassifierHead(latent_dim, num_classes)

        if freeze:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = True

    def forward(self, x):
        return self.classifier(self.encoder(x))
