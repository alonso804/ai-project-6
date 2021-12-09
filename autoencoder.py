import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transform
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import math


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64*2, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(in_features=64*2*7*7, out_features=10)

    def forward(self, image):
        out = F.relu(self.conv1(image))
        out = F.relu(self.conv2(out))
        out = out.view(out.size(0), -1)
        z = self.fc(out)
        return z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(in_features=10, out_features=2*64*7*7)
        self.convTran1 = nn.ConvTranspose2d(
            in_channels=2*64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.convTran2 = nn.ConvTranspose2d(
            in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1)

    def forward(self, latent):
        out = self.fc(latent)
        out = out.view(out.size(0), 64*2, 7, 7)
        out = F.relu(self.convTran1(out))
        out = torch.tanh(self.convTran2(out))
        return out


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        latent = self.encoder(x)
        x_recon = self.decoder(latent)
        return x_recon
