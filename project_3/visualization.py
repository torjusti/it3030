import numpy as np
import torch
import config
import math
from loaders import get_data_loader
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def check_reconstructions(data_set, auto_encoder, validation_net):
    with torch.no_grad():
        img, labels = data_set.get_full_data_set(training=False)

        # Reconstruct images and convert to TensorFlow format.
        reconstructions = auto_encoder.cpu()(torch.tensor(np.moveaxis(img, 3, 1)).float())
        reconstructions = np.moveaxis(reconstructions.detach().double().numpy(), 1, 3)

        # Compute predictability and accuracy.
        pred, acc = validation_net.check_predictability(data=reconstructions, correct_labels=labels)

        print(f'Predictability: {pred}, accuracy: {acc}')

        # Retreive samples for displaying.
        images, labels = data_set.get_random_batch(training=False, batch_size=config.NUM_IMAGES_SHOW)
        images = torch.tensor(np.moveaxis(images, 3, 1)).float()

        # Reconstruct images.
        reconstruction = auto_encoder(images).squeeze().numpy()

        auto_encoder.to(device)

        fig, ax = plt.subplots(1, 2)

        size = int(math.sqrt(config.NUM_IMAGES_SHOW))

        ax[0].imshow(flatten_images(images.squeeze().numpy(), size, size))
        ax[0].set_title('Input images')

        ax[1].imshow(flatten_images(reconstruction, size, size))
        ax[1].set_title('Reconstructed images')

        plt.show()


def display_image_samples(auto_encoder, validation_net, latent_dim):
    with torch.no_grad():
        total_pred, total_cov = 0, 0

        for i in range(8):
            images = auto_encoder.decoder(torch.randn(256, latent_dim).to(device))
            rolled_images = np.moveaxis(images.cpu().numpy(), 1, 3)

            pred, cov = validation_net.check_predictability(data=rolled_images)
            cov = validation_net.check_class_coverage(data=rolled_images)

            total_pred += pred * 1 / 8
            total_cov += cov * 1 / 8

        print(f'Predictability: {total_pred}, coverage {total_cov}')

        images = auto_encoder.decoder(torch.randn(16, latent_dim).to(device))

        plt.imshow(flatten_images(images.cpu().numpy().squeeze(), 4, 4))
        plt.title('Sampled images')
        plt.show()


def flatten_images(images, rows, cols):
    """ Merge an array of images into one big image, usable for plotting. """
    if len(images.shape) == 4:
        depth, width, height = images.shape[1:]
    else:
        width, height = images.shape[1:]
        depth = None

    if depth:
        image = np.zeros(shape=(rows * height, cols * width, depth))
    else:
        image = np.zeros(shape=(rows * height, cols * width))

    i = 0

    for row in range(rows):
        for col in range(cols):
            if i >= len(images):
                break

            if depth:
                image[row * height : (row + 1) * height, col * width : (col + 1) * width, :] = \
                    np.rollaxis(images[i], 0, 3)
            else:
                image[row * height : (row + 1) * height, col * width : (col + 1) * width] = images[i]

            i += 1

    return image
