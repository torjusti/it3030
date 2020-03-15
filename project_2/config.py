from torch.optim import Adam

# Factor deciding split between D1 and D2.
split_factor = 0.8
# Percentage of D1 data used for training versus validation.
d1_train_val_split = 0.8
# Percentage of D2 training data to use for training versus validation.
d2_train_val_split = 0.8
# Percentage of D2 data used for training versus testing.
train_test_split = 0.8
# Number of latent dimensions.
latent_dim = 16
# Learning rate for the latent classifier.
latent_classifier_lr=0.001
# Whether or not the classifier autoencoder should be frozen.
latent_classifier_freeze=True
# Learning rate for the standard classifier.
normal_classifier_lr=0.001
# Learning rate for the autoencoder.
autoencoder_lr=0.001
# Batch size to use. Currently, the same value is used everywhere.
batch_size=128
# Number of epochs to train the autoencoder for.
autoencoder_epochs=15
# Number of epochs to train the latent classifier for.
latent_classifier_epochs=15
# Number of epochs to train the normal classifier for.
normal_classifier_epochs=15
# A flag indicating whether or not tSNE plots of latent
# vectors will be shown at the 3 stages of training.
# Note that freezing of weights affects the last plot.
shown_tsne=True
# Optimizer to use for the autoencoder.
autoencoder_optim=Adam
# Optimizer to use for the latent classifier.
latent_classifier_optim=Adam
# Optimizer to use for the normal classifier.
normal_classifier_optim=Adam
# Number of autoencoder reconstructions to show.
num_reconstructions=16
# The dataset to use. `mnist`, `fmnist`, 'kmnist', 'cifar', 'emnist'.
dataset='cifar'
