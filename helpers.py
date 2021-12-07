import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt

from CustomDataset import ImgDataset


def show(out, title=''):
    print(title)
    out = out.permute(1, 0, 2, 3)
    grilla = torchvision.utils.make_grid(out, 10, 5)
    plt.imshow(transforms.ToPILImage()(grilla), 'jet')
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
            transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
        ]))

    val_dataset = ImgDataset(
        csv_file=val_path,
        root_dir=root_dir,
        transform=torchvision.transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
        ]))

    val_size = int(7 / 10 * val_size)

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
