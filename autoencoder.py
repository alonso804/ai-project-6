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
        self.f1 = nn.Linear(28 * 28 * 4, 28 * 28)
        self.f2 = nn.Linear(28 * 28, 196)
        self.f3 = nn.Linear(196, 49)
        self.f4 = nn.Linear(49, 7)

    def forward(self, image):
        out = F.relu(self.f1(image))
        out = F.relu(self.f2(out))
        out = F.relu(self.f3(out))
        out = F.relu(self.f4(out))
        z = F.relu(out)
        return z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.f1 = nn.Linear(7, 49)
        self.f2 = nn.Linear(49, 196)
        self.f3 = nn.Linear(196, 28 * 28)
        self.f4 = nn.Linear(28 * 28, 28 * 28 * 4)

    def forward(self, z):
        out = F.relu(self.f1(z))
        out = F.relu(self.f2(out))
        out = F.relu(self.f3(out))
        out = F.relu(self.f4(out))
        out = torch.tanh(out)
        return out


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, image):
        z = self.encoder(image)
        out = self.decoder(z)
        return out
