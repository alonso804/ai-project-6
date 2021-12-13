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
    val_loss_avg = []
    for epoch in range(epochs):
        train_loss_avg.append(0)
        val_loss_avg.append(0)
        num_batches = 0
        num_batches_val = 0

        for img, y_img in train_loader:
            y_img = y_img.to(device)
            img = img.to(device)

            img_recon = model(img)
            loss = loss_fn(img_recon, y_img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_avg[-1] += loss.item()
            num_batches += 1

        for img, y_img in val_loader:
            y_img = y_img.to(device)
            img = img.to(device)

            img_recon = model(img)
            loss = loss_fn(img_recon, y_img)

            val_loss_avg[-1] += loss.item()
            num_batches_val += 1

        val_loss_avg[-1] /= num_batches_val
        train_loss_avg[-1] /= num_batches
        print('Epoch [%d / %d] train error: %f - val error: %f' %
              (epoch+1, epochs, train_loss_avg[-1], val_loss_avg[-1]))

    return train_loss_avg, val_loss_avg


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_set, val_set, test_set = get_sets('./train.csv', './val.csv')

    train_loader, val_loader, test_loader = get_loaders(
        train_set,
        val_set,
        test_set
    )

    img, y_img = train_set[10]
    print(img.size())

    show(img)
    show(y_img)

    learning_rate = 0.001
    epochs = 20

    autoencoder = Autoencoder().to(device)

    loss = nn.MSELoss()

    optimizer = torch.optim.Adam(
        params=autoencoder.parameters(), lr=learning_rate, weight_decay=1e-5)

    autoencoder.train()

    loss_result = train(autoencoder, train_loader,
                        val_loader, epochs, loss, optimizer, device)

    torch.save(autoencoder.state_dict(), "./Results/autoencoder-unet.mdl")


if __name__ == "__main__":
    main()
