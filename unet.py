import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transform
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import math


def conv(in_channels, out_channels, kernel=3):
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
            kernel_size=3
        )

    def forward(self, image):
        # Encoder
        print('DEB:', image.shape)
        print('DEB:', self.convd1)
        x1 = self.convd1(image)
        print('DEB:', x1.shape)
        x2 = self.max_pooling(x1)
        x3 = self.convd2(x2)
        x4 = self.max_pooling(x3)
        x5 = self.convd3(x4)
        x6 = self.max_pooling(x5)
        x7 = self.convd4(x6)
        x8 = self.max_pooling(x7)
        x9 = self.convd5(x8)

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

        return out
