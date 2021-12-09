import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
import numpy as np
import math

from CustomDataset import ImgDataset


def UnNormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def show_img(imgs, name, size=3, color=True):
    color_m = 'jet'
    if color == False:
        color_m = 'gray'
    print('*******************' + name + '*********************')
    img_numbers = imgs.shape[0]
    rows = cols = math.ceil(np.sqrt(img_numbers))

    fig = plt.figure(figsize=(rows * size, cols * size))
    for i in range(0, rows * cols):
        fig.add_subplot(rows, cols, i + 1)
        if i < img_numbers:
            plt.imshow(imgs[i].detach())
        plt.show()


def show(out, title=''):
    print(title)
    # plt.imshow(out.permute(1, 2, 0))
    plt.imshow(UnNormalize(out, (0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)).permute(
        1, 2, 0))
    # plt.imshow((UnNormalize(out, (0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)).permute(
    # 1, 2, 0).numpy() * 255).astype(np.uint8))

    # out = out.permute(0, 1, 2)
    # grilla = torchvision.utils.make_grid(out, 10, 5)
    # plt.imshow(transforms.ToPILImage()(grilla).convert('RGB'), 'jet')
    plt.show()


def show_weight(out):
    grilla = torchvision.utils.make_grid(out)
    plt.imshow(transforms.ToPILImage()(grilla), 'jet')
    plt.show()


def get_sets(train_path, val_path, root_dir='./'):
    train_set = ImgDataset(
        csv_file=train_path,
        root_dir=root_dir,
        transform=torchvision.transforms.Compose([
            # transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]))

    val_dataset = ImgDataset(
        csv_file=val_path,
        root_dir=root_dir,
        transform=torchvision.transforms.Compose([
            # transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]))

    val_size = int(7 / 10 * len(val_dataset))

    val_set, test_set = random_split(
        val_dataset,
        [val_size, len(val_dataset) - val_size])

    return train_set, val_set, test_set


def get_loaders(train_set, val_set, test_set, batch_size=64):
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True)

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False)

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False)

    return train_loader, val_loader, test_loader
