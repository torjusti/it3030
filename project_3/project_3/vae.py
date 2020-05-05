import torch
from stacked_mnist import StackedMNISTData, DataMode
from verification_net import VerificationNet
from visualization import check_reconstructions, display_image_samples, flatten_images
from loaders import get_data_loader
import matplotlib.pyplot as plt
import math
import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(torch.nn.Module):
    def __init__(self, num_channels, latent_dim):
        """ Encoder part of the variational auto-encoder. """
        super().__init__()

        self.body = torch.nn.Sequential(
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

            torch.nn.Linear(7 * 7 * 256, 128),
        )

        # Separate heads for the mean and standard deviation.
        self.mean_head = torch.nn.Linear(128, latent_dim)
        self.log_std_head = torch.nn.Linear(128, latent_dim)

    def forward(self, x):
        """ Pass image `x` through the network, predicting mean and std of distribution. """
        # Compute value of last hidden layer.
        x = self.body(x)
        # Compute mean and log-std values.
        return self.mean_head(x), self.log_std_head(x)

    def sample_encoding(self, x):
        """ Sample an encoding for `x`. """
        # Find mean and log-std for samples.
        mean, log_std = self.forward(x)
        # Create distribution with this mean and std.
        normal = torch.distributions.Normal(mean, log_std.exp())
        # Sample using reparameterization trick.
        return normal.rsample(), mean, log_std


class Decoder(torch.nn.Module):
    def __init__(self, num_channels, latent_dim):
        """ Decoder network for the variational auto-encoder. """
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
        """ Find probabilities for output pixels being turned on, given sampled encoding.
        The output stochastic variable X is assumed to be multivariate Bernoulli. """
        x = self.linear_layers(x)
        x = x.view((x.shape[0], 256, 7, 7))
        return self.conv_layers(x)


class VariationalAutoEncoder(torch.nn.Module):
    def __init__(self, num_channels, latent_dim):
        """ Variational auto-encoder network. """
        super().__init__()

        self.encoder = Encoder(num_channels, latent_dim)
        self.decoder = Decoder(num_channels, latent_dim)

    def forward(self, x, return_parameters=False):
        """ Perform the auto-encoding operation on the input. """
        encoding, mean, log_std = self.encoder.sample_encoding(x)

        if return_parameters:
            return self.decoder(encoding), mean, log_std

        return self.decoder(encoding)

    @staticmethod
    def loss_fn(images, reconstruction, mean, log_std):
        # Log likelihood when using multivariate Bernoulli distribution.
        # We sum over all three axes of the reconstructed images. The
        # resulting tensor contains losses for each individual sample.
        reconstruction_loss = torch.sum((images * torch.log(reconstruction + 1e-9)) + \
            (1 - images) * torch.log(1 - reconstruction + 1e-9), axis=(1, 2, 3))

        # Analytical negative KL-divergence between hidden layer representation and unit Gaussian.
        kld = 0.5 * torch.sum(1 + torch.log(log_std.exp() ** 2 + 1e-9) - mean.pow(2) - log_std.exp() ** 2, axis=1)

        # Take mean and convert to gradient descent formulation.
        return -1 * (kld + reconstruction_loss).mean() / images.shape[0]


def main():
    # Create data loader objects.
    binary_complete_mode = DataMode.MONO_BINARY_COMPLETE if config.MODE == 'mono' else DataMode.COLOR_BINARY_COMPLETE
    binary_complete = StackedMNISTData(mode=binary_complete_mode)

    binary_missing_mode = DataMode.MONO_BINARY_MISSING if config.MODE == 'mono' else DataMode.COLOR_BINARY_MISSING
    binary_missing = StackedMNISTData(mode=binary_missing_mode)

    # Create and train verification net.
    net = VerificationNet(force_learn=False)
    net.train(generator=binary_complete, epochs=5)

    # Create PyTorch data loaders.
    bc_train_loader = get_data_loader(binary_complete)
    bm_train_loader = get_data_loader(binary_missing)
    bm_test_loader = get_data_loader(binary_missing, training=False)

    num_channels = 1 if config.MODE == 'mono' else 3

    vae = VariationalAutoEncoder(num_channels, config.LATENT_DIM).to(device)
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=config.VAE_LR)

    if config.LOAD_VAE:
        vae = torch.load(config.VAE_FILE)
    else:
        for _ in range(config.VAE_EPOCHS):
            total_loss, num_batches = 0, 0

            for images, labels in bc_train_loader:
                images = images.float().to(device)

                reconstruction, mean, log_std = vae(images, return_parameters=True)

                loss = vae.loss_fn(images, reconstruction, mean, log_std)

                total_loss += loss
                num_batches += 1

                vae_optimizer.zero_grad()
                loss.backward()
                vae_optimizer.step()

            print(f'Loss: {total_loss / num_batches}')

        torch.save(vae, config.VAE_FILE)

    print('Finished training normal VAE')
    check_reconstructions(binary_complete, vae, net)
    display_image_samples(vae, net, config.LATENT_DIM)

    anomaly_vae = VariationalAutoEncoder(num_channels, config.LATENT_DIM).to(device)
    anomaly_vae_optimizer = torch.optim.Adam(anomaly_vae.parameters(), lr=config.VAE_LR)

    if config.LOAD_ANOMALY_VAE:
        anomaly_vae = torch.load(config.ANOMALY_VAE_FILE)
    else:
        for _ in range(config.ANOMALY_VAE_EPOCHS):
            total_loss, num_batches = 0, 0

            for images, labels in bm_train_loader:
                images = images.float().to(device)

                reconstruction, mean, log_std = anomaly_vae(images, return_parameters=True)

                loss = anomaly_vae.loss_fn(images, reconstruction, mean, log_std)

                total_loss += loss
                num_batches += 1

                anomaly_vae_optimizer.zero_grad()
                loss.backward()
                anomaly_vae_optimizer.step()

            print(f'Loss: {total_loss / num_batches}')

        torch.save(anomaly_vae, config.ANOMALY_VAE_FILE)

    print('Finished training anomaly VAE')
    check_reconstructions(binary_complete, vae, net)
    display_image_samples(vae, net, config.LATENT_DIM)

    with torch.no_grad():
        all_images = None
        all_reconstructions = None
        loss = None

        for images, labels in bm_test_loader:
            images = images.float().to(device)

            reconstructions = anomaly_vae(images)

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
