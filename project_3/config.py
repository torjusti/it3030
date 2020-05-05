# Size of latent dimensions to use.
LATENT_DIM = 64
# Whether or not Wasserstein loss should be used for the GAN.
WASSERSTEIN = False
# Epochs to train GAN for.
GAN_EPOCHS = 500
# GAN discriminator LR.
DISCRIMINATOR_LR = 1e-5
# Size of GAN noise vector.
NOISE_DIM = 32
# Gan generator LR
GENERATOR_LR = 1e-4
# Delay for generator training to use with Wasserstein.
GENERATOR_DELAY = 1
# The number of reconstructed imaged to show.
NUM_IMAGES_SHOW = 36
# Number of epochs to train autoencoder for.
AUTOENCODER_EPOCHS = 25
# Number of epochs to train anomaly autoencoder for.
ANOMALY_AUTOENCODER_EPOCHS = 25
# Number of bad reconstructions to show.
NUM_WORST = 36
# Mode to use, `color` or `mono`.
MODE = 'color'
# LR for auto-encoder.
AUTOENCODER_LR = 1e-3
# Number of epochs to train VAE for.
VAE_EPOCHS = 75
# Number of epochs to train anomaly VAE for.
ANOMALY_VAE_EPOCHS = 75
# LR for VAE training.
VAE_LR = 1e-3
# Load autoencoder from file.
LOAD_AUTOENCODER = True
# File to load from.
AUTOENCODER_FILE = f'{MODE}-autoencoder.pt'
# Load anomaly autoencoder.
LOAD_ANOMALY_AUTOENCODER = True
# Anomaly autoencoder file.
ANOMALY_AUTOENCODER_FILE = f'{MODE}-anomaly-autoencoder.pt'
# Load anomaly VAE from file.
LOAD_ANOMALY_VAE = True
# Load VAE from file.
LOAD_VAE = True
# Anomaly VAE file.
ANOMALY_VAE_FILE = f'{MODE}-anomaly-vae.pt'
# VAE file.
VAE_FILE = f'{MODE}-vae.pt'
# Load generator from file.
LOAD_GENERATOR = True
# Generator file.
GENERATOR_FILE = f'{MODE}-gan.pt'
# Load VEEGAN from file.
LOAD_VEEGAN = True
# File to load VEEGAN generator from.
VEEGAN_GENERATOR_FILE = f'{MODE}-veegan.pt'
