import torch


class Classifier(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3),
            torch.nn.MaxPool2d(2),
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
