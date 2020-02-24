import torch
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from autoencoder import AutoEncoder
from latent_classifier import LatentClassifier
from classifier import Classifier
from utils import compute_accuracy, flatten_images
import config

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def main():
    # Preprocessing applied to all samples.
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.ToTensor(),
    ])

    # Load the dataset.
    train_set = torchvision.datasets.MNIST('data/mnist', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST('data/mnist', train=False, download=True, transform=transform)
    dataset = torch.utils.data.ConcatDataset([train_set, test_set])

    # Find dimensions of the samples in the dataset.
    input_dim = dataset[0][0].shape
    num_classes = 1 + torch.max(train_set.targets).item()

    # Split dataset into D1 and D2.
    d1, d2 = torch.utils.data.random_split(dataset, [round(len(dataset) * config.split_factor),
                                           round(len(dataset) * (1 - config.split_factor))])

    # Split D2 into training and validation datasets.
    d2_train, d2_val = torch.utils.data.random_split(d2, [round(len(d2) * config.train_test_split),
                                                     round(len(d2) * (1 - config.train_test_split))])

    # Create data loaders.
    d1_loader = torch.utils.data.DataLoader(d1, batch_size=config.batch_size, shuffle=True)
    d2_train_loader = torch.utils.data.DataLoader(d2_train, batch_size=config.batch_size, shuffle=True)
    d2_val_loader = torch.utils.data.DataLoader(d2_val, batch_size=config.batch_size, shuffle=True)

    # Create the autoencoder model.
    autoencoder = AutoEncoder(config.latent_dim).to(device)

    autoencoder_loss = torch.nn.MSELoss(reduction='mean')
    autoencoder_optimizer = torch.optim.Adam(autoencoder.parameters(),
                                          lr=config.autoencoder_lr)

    # Train the autoencoder.
    for epoch in range(config.autoencoder_epochs):
        for images, _ in tqdm(d1_loader, desc=f'AutoEncoder - epoch {epoch}'):
            images = images.to(device)
            reconstruction = autoencoder(images)
            loss = autoencoder_loss(reconstruction, images)
            autoencoder_optimizer.zero_grad()
            loss.backward()
            autoencoder_optimizer.step()

    # Display input images and their reconstructoins.
    with torch.no_grad():
        # Get one batch of images and labels.
        images, labels = iter(d2_val_loader).next()

        reconstruction = autoencoder(images[:16].to(device)).cpu().squeeze().numpy()

        fig, ax = plt.subplots(1, 2)

        ax[0].imshow(flatten_images(images[:16].squeeze(), 4, 4))
        ax[0].set_title('Input images')

        ax[1].imshow(flatten_images(reconstruction, 4, 4))
        ax[1].set_title('Reconstructed images')

        plt.show()

    # Create the latent classifier model.
    latent_classifier = LatentClassifier(autoencoder.encoder, config.latent_dim, num_classes,
                                         freeze=config.latent_classifier_freeze).to(device)

    latent_classifier_loss = torch.nn.CrossEntropyLoss()
    latent_classifier_optimizer = torch.optim.Adam(latent_classifier.parameters(),
                                                   lr=config.latent_classifier_lr)

    # Train the latent classifier.
    for epoch in range(config.latent_classifier_epochs):
        for images, labels in tqdm(d2_train_loader, desc=f'Latent classifier - epoch {epoch}'):
            activation = latent_classifier(images.to(device))
            loss = latent_classifier_loss(activation, labels.to(device))
            latent_classifier_optimizer.zero_grad()
            loss.backward()
            latent_classifier_optimizer.step()

    # Create the standard classifier.
    classifier = Classifier(num_classes).to(device)

    classifier_loss = torch.nn.CrossEntropyLoss()
    classifier_optimizer = torch.optim.Adam(classifier.parameters(),
                                            lr=config.normal_classifier_lr)

    # Train the standard classifier.
    for epoch in range(config.normal_classifier_epochs):
        for images, labels in tqdm(d2_train_loader, desc=f'Normal classifier - epoch {epoch}'):
            activation = classifier(images.to(device))
            loss = classifier_loss(activation, labels.to(device))
            classifier_optimizer.zero_grad()
            loss.backward()
            classifier_optimizer.step()

    print(f'Accuracy on D2 - FC using latent classfier: {compute_accuracy(latent_classifier, d2_val_loader, device=device)}')
    print(f'Accuracy on D2 - FC using normal classfier: {compute_accuracy(classifier, d2_val_loader, device=device)}')
    print(f'Accuracy on D1 using latent classfier: {compute_accuracy(latent_classifier, d1_loader, device=device)}')
    print(f'Accuracy on D1 using normal classfier: {compute_accuracy(classifier, d1_loader, device=device)}')

if __name__ == '__main__':
    main()
