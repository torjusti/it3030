import torch


class Classifier(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim[0], 16, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3),
            torch.nn.AdaptiveMaxPool2d((12, 12)),
            torch.nn.ReLU(),

            torch.nn.Flatten(),
            torch.nn.Linear(12 * 12 * 32, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)
