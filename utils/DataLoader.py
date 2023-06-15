import torch
from torch.utils.data import Dataset


#TODO: batching the BDD dataset for CLIP project. This + list comprehension should accelerate the time execution across the data.
class CustomDataloader(Dataset):
    def __init__(self, imgs):
        self.imgs = imgs


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return imgs[idx]


