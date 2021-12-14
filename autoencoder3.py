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
        out = F.relu(self.f1(image))
        out = F.relu(self.f2(out))
        out = F.relu(self.f3(out))
        out = F.relu(self.f4(out))
        z = F.relu(out)
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
        out = F.relu(self.f1(z))
        out = F.relu(self.f2(out))
        out = F.relu(self.f3(out))
        out = F.relu(self.f4(out))
        out = torch.tanh(out)
        return out

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convd1 = conv(4, 32)
        self.convd2 = conv(32, 64)
        self.convd3 = conv(64, 128)
        self.convd4 = conv(128, 256)
        self.convd5 = conv(256, 512)

        # Decoder
        self.convd_t1 = conv_trans(512, 256)
        self.convd_u1 = conv(512, 256)

        self.convd_t2 = conv_trans(256, 128)
        self.convd_u2 = conv(256, 128)

        self.convd_t3 = conv_trans(128, 64)
        self.convd_u3 = conv(128, 64)

        self.convd_t4 = conv_trans(64, 32)
        self.convd_u4 = conv(64, 32)

        self.out = nn.Conv2d(
            in_channels=32,
            out_channels=4,
            kernel_size=1
        )

    

    def forward(self, image):
        # Encoder
        x1 = self.convd1(image)
        x2 = self.max_pooling(x1)
        x3 = self.convd2(x2)
        x4 = self.max_pooling(x3)
        x5 = self.convd3(x4)
        x6 = self.max_pooling(x5)
        x7 = self.convd4(x4)
        x8 = self.max_pooling(x5)
        x9 = self.convd5(x4)

        # Decoder
        x = self.convd_t1(x9)
        y = crop(x7, x)
        x = self.convd_u1(torch.cat([x, y], 1))

        x = self.convd_t2(x)
        y = crop(x5, x)
        x = self.convd_u2(torch.cat([x, y], 1))

        x = self.convd_t3(x)
        y = crop(x3, x)
        x = self.convd_u3(torch.cat([x, y], 1))

        x = self.convd_t4(x)
        y = crop(x1, x)
        x = self.convd_u4(torch.cat([x, y], 1))

        x = self.out(x)

        return x

def conv(in_channels, out_channels, kernel = 3):
  return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=kernel),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels, out_channels, kernel_size=kernel),
      nn.ReLU(inplace=True)
  )

def conv_trans(in_channels, out_channels, kernel=2, stride=2):
  return nn.ConvTranspose2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel,
                            stride=stride)

def crop(tensor, target_tensor):
  target_size = target_tensor.size()[2]
  tensor_size = tensor.size()[2]

  delta = tensor_size - target_size
  delta //= 2

  return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]    
    
    
    

