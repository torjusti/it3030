import torch
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class LatentClassifier(torch.nn.Module):
    def __init__(self, encoder, latent_dim, num_classes, freeze=False, lr=0.001):
        super().__init__()

        self.encoder = encoder

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, num_classes),
        )

        if freeze:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = True

        self.to(device)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return self.classifier(self.encoder(x))

    def train(self, data_loader, epochs=10):
        for epoch in range(epochs):
            for images, labels in tqdm(data_loader, desc=f'Epoch {epoch}'):
                activation = self.forward(images.to(device))

                loss = self.loss_fn(activation, labels.to(device))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
