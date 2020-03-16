import torch
import torchvision
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import collections
import math
from tqdm import tqdm
from autoencoder import AutoEncoder
from latent_classifier import LatentClassifier
from classifier import Classifier
from utils import compute_accuracy, flatten_images, tsne_plot
import config

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def main():
    # Preprocessing applied to all samples. I try to assume as little
    # as possible about the image dimensions and colors, but the
    # output size is going to be >= 28 pixels due to the shapes of
    # the selected convolution and transposed convolution operators.
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    if config.dataset == 'mnist':
        dataset = torchvision.datasets.MNIST
    elif config.dataset == 'fmnist':
        dataset = torchvision.datasets.FashionMNIST
    elif config.dataset == 'cifar':
        dataset = torchvision.datasets.CIFAR10
    elif config.dataset == 'kmnist':
        dataset = torchvision.datasets.KMNIST
    elif config.dataset == 'emnist':
        dataset = torchvision.datasets.EMNIST

    # Load the dataset.
    if config.dataset == 'emnist':
        # For EMNIST, we need to specify which split to use.
        train_set = dataset('data', train=True, download=True, transform=transform, split='letters')
        test_set = dataset('data', train=False, download=True, transform=transform, split='letters')
    else:
        train_set = dataset('data', train=True, download=True, transform=transform)
        test_set = dataset('data', train=False, download=True, transform=transform)

    dataset = torch.utils.data.ConcatDataset([train_set, test_set])

    # Find size of images in the dataset.
    input_dim = dataset[0][0].shape
    # Find number of classes in dataset.
    if isinstance(train_set.targets, list):
        num_classes = 1 + max(train_set.targets)
    else:
        num_classes = 1 + torch.max(train_set.targets).item()

    # Split dataset into D1 and D2.
    d1_indices, d2_indices = train_test_split(list(range(len(dataset))), train_size=config.split_factor,
                                              stratify=[label for _, label in dataset], shuffle=True)
    d1 = torch.utils.data.Subset(dataset, d1_indices)
    d2 = torch.utils.data.Subset(dataset, d2_indices)

    # Split D1 into training and validation sets.  The entirety of D1 is used
    # as a training set since no labels are available. We therefore only split
    # it into a training set and a validation set.
    d1_train_indices, d1_val_indices = train_test_split(list(range(len(d1))),
                                                        train_size=config.d1_train_val_split,
                                                        stratify=[label for _, label in d1])
    d1_train = torch.utils.data.Subset(d1, d1_train_indices)
    d1_val = torch.utils.data.Subset(d1, d1_val_indices)

    # Create data loaders for D1.
    d1_train_loader = torch.utils.data.DataLoader(d1_train, batch_size=config.batch_size, shuffle=True)
    d1_val_loader = torch.utils.data.DataLoader(d1_val, batch_size=config.batch_size, shuffle=True)
    # Data loader for the entire dataset, for testing classifiers on D1.
    d1_loader = torch.utils.data.DataLoader(d1, batch_size=config.batch_size, shuffle=True)

    # Split D2 into training and testing datasets.
    d2_train_indices, d2_test_indices = train_test_split(list(range(len(d2))),
                                                         train_size=config.train_test_split,
                                                         stratify=[label for _, label in d2])

    # Split testing dataset into testing and validation datasets.
    d2_train = torch.utils.data.Subset(d2, d2_train_indices)
    d2_train_indices, d2_val_indices = train_test_split(list(range(len(d2_train))),
                                                        train_size=config.d2_train_val_split,
                                                        stratify=[label for _, label in d2_train])
    d2_train = torch.utils.data.Subset(d2, d2_train_indices)
    d2_val = torch.utils.data.Subset(d2, d2_val_indices)
    d2_test = torch.utils.data.Subset(d2, d2_test_indices)

    # Create data loaders for D2.
    d2_train_loader = torch.utils.data.DataLoader(d2_train, batch_size=config.batch_size, shuffle=True)
    d2_val_loader = torch.utils.data.DataLoader(d2_val, batch_size=config.batch_size, shuffle=True)
    d2_test_loader = torch.utils.data.DataLoader(d2_test, batch_size=config.batch_size, shuffle=True)

    # -----------
    # Autoencoder
    # -----------

    # Create the autoencoder model.
    autoencoder = AutoEncoder(input_dim, config.latent_dim).to(device)

    autoencoder_loss = torch.nn.MSELoss(reduction='mean')
    autoencoder_optimizer = config.autoencoder_optim(autoencoder.parameters(),
                                                     lr=config.autoencoder_lr)

    tsne_plot(autoencoder.encoder, d1_val_loader, 'TSNE plot before training')

    # Information about autoencoder training.
    autoencoder_train_loss, autoencoder_val_loss = [], []

    # Train the autoencoder.
    for epoch in range(config.autoencoder_epochs):
        total_train_examples, total_val_examples = 0, 0
        total_train_loss, total_val_loss = 0, 0

        for images, _ in tqdm(d1_train_loader, desc=f'AutoEncoder - epoch {epoch}'):
            images = images.to(device)
            reconstruction = autoencoder(images)
            loss = autoencoder_loss(reconstruction, images)
            autoencoder_optimizer.zero_grad()
            loss.backward()
            autoencoder_optimizer.step()
            total_train_loss += loss
            total_train_examples += images.shape[0]

        autoencoder.eval()

        # Test on validation set.
        with torch.no_grad():
            for val_images, _ in d1_val_loader:
                val_images = val_images.to(device)
                reconstruction = autoencoder(val_images)
                total_val_loss += autoencoder_loss(reconstruction, val_images)
                total_val_examples += val_images.shape[0]

        autoencoder.train()

        autoencoder_train_loss.append(total_train_loss / total_train_examples)
        autoencoder_val_loss.append(total_val_loss / total_val_examples)

    plt.plot(autoencoder_train_loss, 'b', label='Autoencoder training loss')
    plt.plot(autoencoder_val_loss, 'r', label='Autoencoder validation loss')
    plt.legend()
    plt.title('Autoencoder loss')
    plt.show()

    # Display input images and their reconstructions.
    with torch.no_grad():
        # Get one batch of images and labels.
        images, labels = iter(d2_test_loader).next()

        reconstruction = autoencoder(images[:config.num_reconstructions].to(device)).cpu().squeeze().numpy()

        fig, ax = plt.subplots(1, 2)

        size = math.ceil(math.sqrt(config.num_reconstructions))

        ax[0].imshow(flatten_images(images[:config.num_reconstructions].cpu().squeeze().numpy(), size, size))
        ax[0].set_title('Input images')

        ax[1].imshow(flatten_images(reconstruction, size, size))
        ax[1].set_title('Reconstructed images')

        plt.show()

    tsne_plot(autoencoder.encoder, d1_val_loader, 'TSNE plot after training autoencoder')

    # --------------------------
    # Semi-supervised classifier
    # --------------------------

    # Create the latent classifier model.
    latent_classifier = LatentClassifier(autoencoder.encoder, config.latent_dim, num_classes,
                                         freeze=config.latent_classifier_freeze).to(device)

    latent_classifier_loss = torch.nn.CrossEntropyLoss()
    latent_classifier_optimizer = config.latent_classifier_optim(latent_classifier.parameters(),
                                                                 lr=config.latent_classifier_lr)

    # Statistics for the latent classifier.
    latent_classifier_train_accuracy = []
    latent_classifier_val_accuracy = []

    # Train the latent classifier.
    for epoch in range(config.latent_classifier_epochs):
        total_train_samples = 0
        total_train_correct = 0

        for images, labels in tqdm(d2_train_loader, desc=f'Latent classifier - epoch {epoch}'):
            activation = latent_classifier(images.to(device))
            labels = labels.to(device)
            loss = latent_classifier_loss(activation, labels)
            latent_classifier_optimizer.zero_grad()
            loss.backward()
            latent_classifier_optimizer.step()

            total_train_correct += (torch.max(activation, 1)[1] == labels).float().sum()
            total_train_samples += images.shape[0]

        latent_classifier_train_accuracy.append(total_train_correct / total_train_samples)
        latent_classifier_val_accuracy.append(compute_accuracy(latent_classifier, d2_val_loader, device=device))

    tsne_plot(latent_classifier.encoder, d1_val_loader, 'TSNE plot after training classifier')

    # -------------------
    # Standard classifier
    # -------------------

    # Statistics for the standard classifier.
    classifier_train_accuracy = []
    classifier_val_accuracy = []

    # Create the standard classifier.
    classifier = Classifier(input_dim, num_classes).to(device)

    classifier_loss = torch.nn.CrossEntropyLoss()
    classifier_optimizer = config.normal_classifier_optim(classifier.parameters(),
                                            lr=config.normal_classifier_lr)

    # Train the standard classifier.
    for epoch in range(config.normal_classifier_epochs):
        total_train_samples = 0
        total_train_correct = 0

        for images, labels in tqdm(d2_train_loader, desc=f'Normal classifier - epoch {epoch}'):
            activation = classifier(images.to(device))
            labels = labels.to(device)
            loss = classifier_loss(activation, labels)
            classifier_optimizer.zero_grad()
            loss.backward()
            classifier_optimizer.step()

            total_train_correct += (torch.max(activation, 1)[1] == labels).float().sum()
            total_train_samples += images.shape[0]

        classifier_train_accuracy.append(total_train_correct / total_train_samples)
        classifier_val_accuracy.append(compute_accuracy(classifier, d2_val_loader, device=device))

    # -----------
    # Final plots
    # -----------

    plt.plot(latent_classifier_train_accuracy, label='Semi-supervised training accuracy')
    plt.plot(latent_classifier_val_accuracy, label='Semi-supervised validation accuracy')
    plt.plot(classifier_train_accuracy, label='Supervised training accuracy')
    plt.plot(classifier_val_accuracy, label='Supervised validation accuracy')
    plt.title('Comparative Classifier Learning')
    plt.legend()
    plt.show()

    plt.show()

    print(f'Accuracy on D2 - FC using latent classfier: {compute_accuracy(latent_classifier, d2_test_loader, device=device)}')
    print(f'Accuracy on D2 - FC using normal classfier: {compute_accuracy(classifier, d2_test_loader, device=device)}')
    print(f'Accuracy on D1 using latent classfier: {compute_accuracy(latent_classifier, d1_loader, device=device)}')
    print(f'Accuracy on D1 using normal classfier: {compute_accuracy(classifier, d1_loader, device=device)}')


if __name__ == '__main__':
    main()
