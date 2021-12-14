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
        self.f1 = nn.Conv2d(4, 16, 3, stride=2, padding=1)  # 128
        self.f2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # 64
        self.f3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 32
        self.f4 = nn.Conv2d(64, 128, 32)

    def forward(self, image):
        out = F.tanh(self.f1(image))
        #print('out1:', out.shape)
        out = F.tanh(self.f2(out))
        #print('out2:', out.shape)
        out = F.tanh(self.f3(out))
        #print('out3:', out.shape)
        out = F.tanh(self.f4(out))
        z = F.tanh(out)
        return z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.f1 = nn.ConvTranspose2d(128, 64, 32)
        self.f2 = nn.ConvTranspose2d(
            64, 32, 3, stride=2, padding=1, output_padding=1)
        self.f3 = nn.ConvTranspose2d(
            32, 16, 3, stride=2, padding=1, output_padding=1)
        self.f4 = nn.ConvTranspose2d(
            16, 4, 3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        out = F.tanh(self.f1(z))
        out = F.tanh(self.f2(out))
        out = F.tanh(self.f3(out))
        out = F.tanh(self.f4(out))
        out = F.tanh(out)
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
