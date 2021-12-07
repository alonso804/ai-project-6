import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io


class ImgDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)

        y_img_path = os.path.join(
            self.root_dir, self.annotations.iloc[index, 1])
        y_image = io.imread(y_img_path)

        if self.transform:
            image = self.transform(image)
            y_image = self.transform(y_image)

        return (image, y_image)
