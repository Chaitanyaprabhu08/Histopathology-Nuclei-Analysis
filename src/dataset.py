import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class NucleiDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(".png")
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]

        image = np.array(
            Image.open(os.path.join(self.image_dir, filename))
        ).astype(np.float32) / 255.0

        mask = np.array(
            Image.open(os.path.join(self.mask_dir, filename))
        ).astype(np.float32)

        image = torch.tensor(image).permute(2, 0, 1)
        mask = torch.tensor(mask).unsqueeze(0)

        return image, mask