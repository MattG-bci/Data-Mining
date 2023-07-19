import os
import warnings
import time
import string

import re
import torch
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from src.search_frames import search_frames, load_model
from src.visualisation import visualise_top_images
from src.text_preprocessing import process_caption
from utils.CustomDataset import CustomDataset
from utils.process_files import concat_configured_parquets
warnings.filterwarnings("ignore")


config_path = "./config/config_files"
assert len(os.listdir(config_path)) > 0, \
    "No configuration files provided. Please execute config.sh file first to configure your data into parquet files."

caption = input("\nWhat is your keyword?: ")
n_matches = int(input("How many top matches would you like to see?: "))
caption = process_caption(caption)
df = concat_configured_parquets(config_path)

n_samples = int(input(f"How many samples do you wish to consider (max: {len(df)})?: "))
print("\n################ PROCESSING - PLEASE WAIT ################")

df = df.iloc[0:n_samples, :]
images = df.name.values

def _convert_image_to_rgb(image):
    return image.convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop((224, 224)),
    _convert_image_to_rgb,
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

dataset = CustomDataset(images, transform)
dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running on {device.upper()}...")
model, _ = load_model("ViT-B/16", device)
logits = np.array([])
start = time.time()
for i_batch, sample_batched in enumerate(dataloader):
    logits_img = search_frames(sample_batched, caption, model, device)
    logits = np.concatenate((logits, logits_img), axis=None)

print(f"Time elapsed: {(time.time() - start):.4f} seconds")
ids = ((-logits).argsort()[:n_matches]).tolist()
visualise_top_images(images[ids], logits[ids], caption)


