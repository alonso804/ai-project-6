import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

from helpers import show_img, show, show_weight, get_sets, get_loaders
from autoencoder import Autoencoder


def train(model, train_loader, val_loader, epochs, loss_fn, optimizer, device):
    train_loss_avg = []
    for epoch in range(epochs):
        train_loss_avg.append(0)
        num_batches = 0

        for image_batch, _ in train_loader:
            image_batch = image_batch.to(device)
          #  print(image_batch.size())

            image_batch_recon = model(image_batch)
           # print(image_batch_recon.size())
            loss = loss_fn(image_batch_recon, image_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_avg[-1] += loss.item()
            num_batches += 1

        train_loss_avg[-1] /= num_batches
        print('Epoch [%d / %d] average reconstruction error: %f' %
              (epoch+1, epochs, train_loss_avg[-1]))
    return train_loss_avg


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_set, val_set, test_set = get_sets('./train.csv', './val.csv')

    train_loader, val_loader, test_loader = get_loaders(
        train_set,
        val_set,
        test_set
    )

    img, _ = train_set[0]
    print(img.shape)
    show_img(img, 'Temp')

    capacity = 64
    latent_dims = 10
    learning_rate = 0.001
    epochs = 20

    autoencoder = Autoencoder().to(device)

    loss = nn.MSELoss()

    optimizer = torch.optim.Adam(
        params=autoencoder.parameters(), lr=learning_rate, weight_decay=1e-5)

    autoencoder.train()

    # loss_result = train(autoencoder, train_loader, epochs, loss)

    # torch.save(autoencoder.state_dict(), "./Results/autoencoder.mdl")


if __name__ == "__main__":
    main()
