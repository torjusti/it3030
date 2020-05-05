import torch
import torch.nn.functional as F
from stacked_mnist import StackedMNISTData, DataMode
from verification_net import VerificationNet
from visualization import flatten_images
from loaders import get_data_loader
import matplotlib.pyplot as plt
import numpy as np
import config
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Generator(torch.nn.Module):
    def __init__(self, channels, noise_dim):
        """ Generator part of the Generative Adversarial Network. The role
        of this network is to generate new samples from random noise. """
        super().__init__()

        # Number of elements in the noise vector.
        self.noise_dim = noise_dim

        self.l0 = torch.nn.Linear(noise_dim, 7 * 7 * 256)

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

            torch.nn.ConvTranspose2d(32, channels, 3, padding=1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        """ Returns an image from a given noise vector `x`. """
        x = F.relu(self.l0(x).view(x.shape[0], 256, 7, 7))
        return self.conv_layers(x)


class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        """ Discriminator part of the Generative Adversarial Network. The role
        of this network is to classify input images as either fake or real. """
        super().__init__()

        self.network = torch.nn.Sequential(
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
            torch.nn.Linear(7 * 7 * 256, 1),
        )

    def forward(self, x):
        """ Predict whether or not images `x` are fake. """
        # Note that activations are handled elsewhere.
        return self.network(x)

def show_images(generator, net, epoch=None):
    generator.eval()

    with torch.no_grad():
        total_pred, total_cov = 0, 0

        for i in range(8):
            images = generator(torch.randn(256, config.NOISE_DIM).to(device))

            rolled_images = np.moveaxis(images.cpu().numpy(), 1, 3)

            pred, cov = net.check_predictability(data=rolled_images)
            cov = net.check_class_coverage(data=rolled_images)

            total_pred += pred * 256 / 2048
            total_cov += cov * 256 / 2048

        print(f'Predictability: {total_pred}, coverage {total_cov}')

        images = generator(torch.randn(config.NUM_IMAGES_SHOW, config.NOISE_DIM).to(device))

        size = math.ceil(math.sqrt(config.NUM_IMAGES_SHOW))

        plt.imshow(flatten_images(images.cpu().numpy().squeeze(), size, size))
        plt.title('Sampled images')

        if epoch is not None:
            plt.savefig(f'gan-epoch-{epoch}.png')
        else:
            plt.show()

    generator.train()

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

    generator = Generator(num_channels, config.NOISE_DIM).to(device)
    discriminator = Discriminator(num_channels).to(device)

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=config.GENERATOR_LR, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.DISCRIMINATOR_LR, betas=(0.5, 0.999))

    # Train for a fixed number of epochs.
    if config.LOAD_GENERATOR:
        generator = torch.load(config.GENERATOR_FILE)
        show_images(generator, net)
    else:
        for epoch in range(config.GAN_EPOCHS):
            # Enumerate minibatches to use for delaying the generator updates.
            for i, minibatch in enumerate(fc_train_loader):
                # Ignore labels.
                real_images, _ = minibatch
                # Move images to GPU.
                real_images = real_images.float().to(device)

                # Generate some samples using the generator. Detach, so
                # that no gradients are propagated through the generator.
                fake_images = generator(torch.randn(real_images.shape[0], config.NOISE_DIM).to(device)).detach()

                if config.WASSERSTEIN:
                    real_loss = torch.mean(discriminator(real_images))
                    fake_loss = torch.mean(discriminator(fake_images))
                    # Want discriminator to rate real images high and fakes low.
                    discriminator_loss = fake_loss - real_loss
                else:
                    fake_classifications = discriminator(fake_images)
                    real_classifications = discriminator(real_images)

                    # Softmax is applied implicitly.
                    discriminator_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        torch.cat([fake_classifications, real_classifications], dim=0),

                        # Want discriminator to rate real images 1 and fakes 0.
                        torch.cat([
                            torch.zeros_like(fake_classifications),
                            torch.ones_like(real_classifications)
                        ], dim=0),
                    )

                discriminator_optimizer.zero_grad()
                discriminator_loss.backward()
                discriminator_optimizer.step()

                if config.WASSERSTEIN:
                    for param in discriminator.parameters():
                        param.data.clamp_(-0.01, 0.01)

                if i % config.GENERATOR_DELAY == 0:
                    fake_images = generator(torch.randn(real_images.shape[0], config.NOISE_DIM).to(device))

                    if config.WASSERSTEIN:
                        # Want to force discriminator to rate all these fake images highly.
                        generator_loss = -1 * torch.mean(discriminator(fake_images))
                    else:
                        classifications = discriminator(fake_images)

                        # Want to force discriminator to rate all these fakes as 1.
                        generator_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            classifications, torch.ones_like(classifications))

                    generator_optimizer.zero_grad()
                    generator_loss.backward()
                    generator_optimizer.step()

                if i % 10 == 0:
                    print(f'Generator: {generator_loss}, discriminator: {discriminator_loss}')

            show_images(generator, net, epoch)

            torch.save(generator, config.GENERATOR_FILE)

if __name__ == '__main__':
    main()
