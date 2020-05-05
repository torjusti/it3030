import torch
from stacked_mnist import StackedMNISTData, DataMode
from verification_net import VerificationNet
from loaders import get_data_loader
from gan import Generator, show_images
import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Reconstructor(torch.nn.Module):
    def __init__(self, num_channels, noise_dim):
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28 * 28 * num_channels, 512),
            torch.nn.Linear(512, 256),
            torch.nn.Linear(256, noise_dim),
        )

    def forward(self, x):
        return self.network(x)

class LikelihoodRatioDiscriminator(torch.nn.Module):
    def __init__(self, channels, noise_dim):
        super().__init__()

        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 32, 4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Conv2d(32, 64, 4, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Flatten(),
        )

        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(7 * 7 * 256 + noise_dim, 512),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Linear(512, 1),
        )

    def forward(self, z, x):
        x = self.conv_layers(x)
        return self.linear_layers(torch.cat([x, z], dim=1))


def main():
    # Create data loader objects.
    float_complete_mode = DataMode.MONO_FLOAT_COMPLETE if config.MODE == 'mono' else DataMode.COLOR_FLOAT_COMPLETE
    float_complete = StackedMNISTData(mode=float_complete_mode)

    # Create and train verification net.
    net = VerificationNet(force_learn=False)
    net.train(generator=float_complete, epochs=5)

    # Create PyTorch data loaders.
    fc_train_loader = get_data_loader(float_complete)

    num_channels = 1 if config.MODE == 'mono' else 3

    reconstructor = Reconstructor(num_channels, config.NOISE_DIM).to(device)
    generator = Generator(num_channels, config.NOISE_DIM).to(device)
    discriminator = LikelihoodRatioDiscriminator(num_channels, config.NOISE_DIM).to(device)

    reconstructor_optimizer = torch.optim.Adam(reconstructor.parameters(), lr=config.GENERATOR_LR)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=config.GENERATOR_LR)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.DISCRIMINATOR_LR)

    # Train for a fixed number of epochs.
    if config.LOAD_VEEGAN:
        generator = torch.load(config.VEEGAN_GENERATOR_FILE)
        show_images(generator, net)
    else:
        for epoch in range(config.GAN_EPOCHS):
            # Enumerate minibatches to use for delaying the generator updates.
            for i, minibatch in enumerate(fc_train_loader):
                # Ignore labels.
                real_images, _ = minibatch
                # Move images to GPU.
                real_images = real_images.float().to(device)

                # Draw sample from assumed true generator of compressed images.
                z = torch.randn(real_images.shape[0], config.NOISE_DIM).to(device)

                # Sample generated images using assumed samples.
                sampled_images = generator(z)

                # Reconstruct noise vector from true image.
                reconstructed_noise = reconstructor(real_images)

                # ------------------
                # Discriminator loss
                # ------------------

                discriminator_loss = -torch.sum(torch.log(1e-9 + torch.sigmoid(discriminator(z, sampled_images.detach()))) \
                    + torch.log(1 - torch.sigmoid(discriminator(reconstructed_noise.detach(), real_images) + 1e-9)), dim=1).mean()

                discriminator_optimizer.zero_grad()
                discriminator_loss.backward()
                discriminator_optimizer.step()

                # ------------------
                # Reconstructor loss
                # ------------------

                reconstructor_loss = torch.sum((z - reconstructor(sampled_images.detach())) ** 2).mean()

                reconstructor_optimizer.zero_grad()
                reconstructor_loss.backward()
                reconstructor_optimizer.step()

                # --------------
                # Generator loss
                # --------------

                generator_loss = torch.sum(discriminator(z, sampled_images)).mean() + torch.sum((z - \
                    reconstructor(sampled_images.detach())) ** 2).mean()

                generator_optimizer.zero_grad()
                generator_loss.backward()
                generator_optimizer.step()

                if i % 10 == 0:
                    print(f'Generator: {generator_loss}, discriminator: {discriminator_loss}')

            show_images(generator, net, epoch)

            torch.save(generator, f'{config.VEEGAN_GENERATOR_FILE}-{epoch}')


if __name__ == '__main__':
    main()
