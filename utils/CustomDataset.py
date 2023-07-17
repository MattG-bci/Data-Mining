import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, imgs, transforms):
        self.imgs = imgs
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        image = Image.open(self.imgs[index]).convert("RGB")
        image = self.transforms(image)
        return image


