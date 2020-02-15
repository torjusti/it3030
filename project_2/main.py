import torch
import torchvision
from tqdm import tqdm
import numpy as np
from auto_encoder import AutoEncoder
from latent_classifier import LatentClassifier
from classifier import Classifier
from utils import compute_accuracy
import config


def main():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.MNIST('data/mnist', train=True, download=True, transform=transform)

    input_dim = dataset[0][0].shape
    num_classes = 1 + torch.max(dataset.targets).item()

    d1, d2 = torch.utils.data.random_split(dataset, [round(len(dataset) * config.split_factor),
                                           round(len(dataset) * (1 - config.split_factor))])

    d2_train, d2_val = torch.utils.data.random_split(d2, [round(len(d2) * config.train_test_split),
                                                     round(len(d2) * (1 - config.train_test_split))])

    d1_loader = torch.utils.data.DataLoader(d1, batch_size=config.batch_size, shuffle=True)
    d2_train_loader = torch.utils.data.DataLoader( d2_train, batch_size=config.batch_size, shuffle=True)
    d2_val_loader = torch.utils.data.DataLoader(d2_val, batch_size=config.batch_size, shuffle=True)

    autoencoder = AutoEncoder(input_dim, latent_dim=16, lr=config.autoencoder_lr)
    autoencoder.train(d1_loader, config.autoencoder_epochs)

    latent_classifier = LatentClassifier(autoencoder.encoder, config.latent_dim, num_classes,
        lr=config.latent_classifier_lr, freeze=config.latent_classifier_freeze)

    latent_classifier.train(d2_train_loader, config.latent_classifier_epochs)

    classifier = Classifier(input_dim, num_classes, config.normal_classifier_lr)
    classifier.train(d2_train_loader, config.normal_classifier_epochs)

    print(f'Accuracy on D2 - FC using latent classfier: {compute_accuracy(latent_classifier, d2_val_loader)}')
    print(f'Accuracy on D2 - FC using normal classfier: {compute_accuracy(classifier, d2_val_loader)}')
    print(f'Accuracy on D1 using latent classfier: {compute_accuracy(latent_classifier, d1_loader)}')
    print(f'Accuracy on D1 using normal classfier: {compute_accuracy(classifier, d1_loader)}')

if __name__ == '__main__':
    main()
