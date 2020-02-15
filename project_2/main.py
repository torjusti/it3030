import torch
import torchvision
from tqdm import tqdm

devide = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, latent_dim),
        )

    def forward(self, x):
        return self.network(x)


class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, input_dim):
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, input_dim),
        )

    def forward(self, x):
        return self.network(x)


class AutoEncoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, lr=0.001):
        super().__init__()

        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

        self.loss_fn = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)


    def forward(self, x):
        return self.decoder(self.encoder(x))

    def train(self, data_loader, epochs=1):
        for epoch in range(epochs):
            for images, _ in tqdm(data_loader, desc=f'Epoch {epoch}'):
                # TODO: Figure out how to handle this.
                images = images.reshape((images.shape[0], 28 * 28))

                reconstruction = self.forward(images)

                loss = self.loss_fn(reconstruction, images)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


class LatentClassifier(torch.nn.Module):
    def __init__(self, encoder, latent_dim, num_classes, lr=0.001):
        super().__init__()

        self.encoder = encoder

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, num_classes),
        )

        self.loss_fm = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.adam(self.parameters(), lr=lr)

    def forward(self, x):
        return self.classifier(self.encoder(x))

    def train(self, data_loader, epochs=1):
        for epoch in range(epochs):
            for images, labels in range(epochs):
                images = images.reshape((images.shape[0], 28 * 28))

                activation = self.forward(images)

                loss = self.loss_fn(activation, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


def main():
    split_factor = 0.8
    train_test_split = 0.8
    latent_dim = 16

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.MNIST('data/mnist', train=True, download=True, transform=transform)

    d1, d2 = torch.utils.data.random_split(dataset, [round(len(dataset) * split_factor),
                                           round(len(dataset) * (1 - split_factor))])

    d2_train, d2_val = torch.utils.data.random_split(dataset, round(len(dataset) * train_test_split),
                                                     round(len(dataset) * (1 - train_test_split)))

    d1_loader = torch.utils.data.DataLoader(d1, batch_size=128, shuffle=True)
    d2_train_loader = torch.utils.data.DataLoader(d2_train, batch_size=128, shuffle=True)
    d2_val_loader = torch.utils.data.DataLoader(d2_val, batch_size=128, shuffle=True)

    autoencoder = AutoEncoder(28 * 28, latent_dim=16)

    autoencoder.train(d1_loader)

    classifier = LatentClassifier(autoencoder.encoder, latent_dim, 10)

    classifier.train(d2_train_loader)

if __name__ == '__main__':
    main()
