import torch
from tqdm import tqdm
from encoder import Encoder
from decoder import Decoder

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class AutoEncoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, lr=0.001):
        super().__init__()

        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

        self.to(device)

        self.loss_fn = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def train(self, data_loader, epochs=10):
        for epoch in range(epochs):
            for images, _ in tqdm(data_loader, desc=f'Epoch {epoch}'):
                images = images.to(device)

                reconstruction = self.forward(images)

                loss = self.loss_fn(reconstruction, images)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
