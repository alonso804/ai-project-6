import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io
import PIL.Image


def UnNormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


class ImgDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        # image = np.array(PIL.Image.open(img_path).convert('RGB'))
        # image = np.array(PIL.Image.open(img_path))
        image = io.imread(img_path)
        print(image)
        print('--------------------------')

        y_img_path = os.path.join(
            self.root_dir, self.annotations.iloc[index, 1])
        y_image = io.imread(y_img_path)

        if self.transform:
            image = self.transform(image)
            y_image = self.transform(y_image)

        print((UnNormalize(image, (0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5)).permute(
            1, 2, 0).numpy() * 255).astype(np.uint8))
        return (image, y_image)
