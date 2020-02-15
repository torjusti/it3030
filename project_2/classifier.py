import torch
from tqdm import tqdm
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Classifier(torch.nn.Module):
    def __init__(self, input_dim, num_classes, lr=0.001):
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(np.prod(input_dim), 256),
            torch.nn.Linear(256, 128),
            torch.nn.Linear(128, num_classes),
        )

        self.to(device)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return self.network(x)

    def train(self, data_loader, epochs=10):
        for epoch in range(epochs):
            for images, labels in tqdm(data_loader, desc=f'Epoch {epoch}'):
                activation = self.forward(images.to(device))

                loss = self.loss_fn(activation, labels.to(device))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
