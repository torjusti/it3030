import torch
from stacked_mnist import StackedMNISTData, DataMode
from verification_net import VerificationNet
from visualization import check_reconstructions, display_image_samples, flatten_images
from loaders import get_data_loader
import matplotlib.pyplot as plt
import config
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(torch.nn.Module):
    def __init__(self, num_channels, latent_dim):
        """ Encoder network, used in the auto-encoder. """
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(num_channels, 32, 4, stride=2, padding=1),
            torch.nn.ReLU(),

            torch.nn.Conv2d(32, 64, 4, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),

            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),

            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),

            torch.nn.Flatten(),

            torch.nn.Linear(7 * 7 * 256, latent_dim),
        )

    def forward(self, x):
        """ Compute latent representation for `x`. """
        return self.network(x)


class Decoder(torch.nn.Module):
    def __init__(self, num_channels, latent_dim):
        """ Decoder part of the auto-encoder. """
        super().__init__()

        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 7 * 7 * 256),
            torch.nn.ReLU(),
        )

        self.conv_layers = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),

            torch.nn.ConvTranspose2d(128, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),

            torch.nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),

            torch.nn.ConvTranspose2d(32, num_channels, 3, padding=1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        """ Retrieve shaped image frm latent vector `x`. """
        x = self.linear_layers(x)
        x = x.view((x.shape[0], 256, 7, 7))
        return self.conv_layers(x)


class AutoEncoder(torch.nn.Module):
    def __init__(self, num_channels, latent_dim):
        """ Auto-encoder consisting of an encoder and a decoder. """
        super().__init__()

        self.encoder = Encoder(num_channels, latent_dim)
        self.decoder = Decoder(num_channels, latent_dim)

    def forward(self, x):
        """ Pass image through auto-encoder. """
        return self.decoder(self.encoder(x))


def main():
    # Create data loader objects.
    float_complete_mode = DataMode.MONO_FLOAT_COMPLETE if config.MODE == 'mono' else DataMode.COLOR_FLOAT_COMPLETE
    float_complete = StackedMNISTData(mode=float_complete_mode)

    float_missing_mode = DataMode.MONO_FLOAT_MISSING if config.MODE == 'mono' else DataMode.COLOR_FLOAT_MISSING
    float_missing = StackedMNISTData(mode=float_missing_mode)

    # Create and train verification net.
    net = VerificationNet(force_learn=False)
    net.train(generator=float_complete, epochs=5)

    # Create PyTorch data loaders.
    fc_train_loader = get_data_loader(float_complete)
    fm_train_loader = get_data_loader(float_missing)
    fm_test_loader = get_data_loader(float_missing, training=False)

    num_channels = 1 if config.MODE == 'mono' else 3

    auto_encoder = AutoEncoder(num_channels, config.LATENT_DIM).to(device)
    auto_optimizer = torch.optim.Adam(auto_encoder.parameters(), lr=config.AUTOENCODER_LR)
    loss_fn = torch.nn.MSELoss()

    if config.LOAD_AUTOENCODER:
        auto_encoder = torch.load(config.AUTOENCODER_FILE)
    else:
        for _ in range(config.AUTOENCODER_EPOCHS):
            total_loss, num_batches = 0, 0

            for images, labels in fc_train_loader:
                images = images.float().to(device)

                loss = loss_fn(auto_encoder(images), images)

                total_loss += loss
                num_batches += 1

                auto_optimizer.zero_grad()
                loss.backward()
                auto_optimizer.step()

            print(f'Loss: {total_loss / num_batches}')

        torch.save(auto_encoder, config.AUTOENCODER_FILE)

    auto_encoder.eval()

    print('Finished training normal autoencoder')
    check_reconstructions(float_complete, auto_encoder, net)
    display_image_samples(auto_encoder, net, config.LATENT_DIM)

    anomaly_auto_encoder = AutoEncoder(num_channels, config.LATENT_DIM).to(device)
    anomaly_optimizer = torch.optim.Adam(anomaly_auto_encoder.parameters(), lr=config.AUTOENCODER_LR)

    if config.LOAD_ANOMALY_AUTOENCODER:
        anomaly_auto_encoder = torch.load(config.ANOMALY_AUTOENCODER_FILE)
    else:
        for _ in range(config.ANOMALY_AUTOENCODER_EPOCHS):
            total_loss, num_batches = 0, 0

            for images, labels in fm_train_loader:
                images = images.float().to(device)

                loss = loss_fn(anomaly_auto_encoder(images), images)

                total_loss += loss
                num_batches += 1

                anomaly_optimizer.zero_grad()
                loss.backward()
                anomaly_optimizer.step()

            print(f'Loss: {total_loss / num_batches}')

        torch.save(anomaly_auto_encoder, config.ANOMALY_AUTOENCODER_FILE)

    anomaly_auto_encoder.eval()

    print('Finished training normal autoencoder with missing 8')
    check_reconstructions(float_complete, auto_encoder, net)
    display_image_samples(auto_encoder, net, config.LATENT_DIM)

    with torch.no_grad():
        all_images = None
        all_reconstructions = None
        loss = None

        for images, labels in fm_test_loader:
            images = images.float().to(device)

            reconstructions = anomaly_auto_encoder(images)

            batch_loss = ((images - reconstructions) ** 2).sum(axis=(1, 2, 3)) / 1024

            if loss is None:
                loss = batch_loss.cpu()
            else:
                loss = torch.cat([loss, batch_loss.cpu()])

            if all_images is None:
                all_images = images.cpu()
            else:
                all_images = torch.cat([all_images, images.cpu()])

            if all_reconstructions is None:
                all_reconstructions = reconstructions.cpu()
            else:
                all_reconstructions = torch.cat([all_reconstructions, reconstructions.cpu()])

        worst_indices = loss.numpy().argsort()[-1:-(config.NUM_WORST + 1):-1]

        fig, ax = plt.subplots(1, 2)

        size = math.ceil(math.sqrt(config.NUM_WORST))

        ax[0].imshow(flatten_images(all_images.squeeze().numpy()[worst_indices], size, size))
        ax[0].set_title('Input images')

        ax[1].imshow(flatten_images(all_reconstructions.numpy()[worst_indices].squeeze(), size, size))
        ax[1].set_title('Reconstructed images')

        plt.show()


if __name__ == '__main__':
    main()
