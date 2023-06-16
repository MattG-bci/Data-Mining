import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image


#TODO: batching the BDD dataset for CLIP project. This + list comprehension should accelerate the time execution across the data.
class CustomDataset(Dataset):
    def __init__(self, imgs, tags, transforms):
        self.imgs = imgs
        self.tags = tags
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):

        image = Image.open(self.imgs[index]).convert("RGB")
        image = self.transforms(image)
        tags = self.tags[index]

        return image


