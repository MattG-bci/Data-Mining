import argparse
import os
import numpy as np
import pandas as pd
from src.search_representation import search_frames, load_model
from src.visualisiation import visualise_top_images
from src.metrics import window_precision
from utils.DataLoader import *
from src.structure_data import *
from torch.utils.data import DataLoader
from torchvision import transforms
import warnings
import time

warnings.filterwarnings("ignore")

caption = input("\nWhat is your keyword?: ")
n_matches = int(input("How many top matches you would like to see?: "))

def process_caption(caption):
    if "_" in caption:
        caption = caption.replace("_", " ")
    return caption

caption = process_caption(caption.lower())

train_path = "./src/pq_labels/det_train_new.parquet"
val_path = "./src/pq_labels/det_val_new.parquet"

df_train = concatenate_tags(train_path)
df_val = concatenate_tags(val_path)
df = pd.concat([df_train, df_val])

n_samples = int(input(f"How many samples you wish to consider (max: {len(df)})?: "))
print("\n################ PROCESSING - PLEASE WAIT ################")

df = df.iloc[0:n_samples, :]
images = df.name.values
tags = df.iloc[:, 1:].values

def _convert_image_to_rgb(image):
    return image.convert("RGB")

transform = transforms.Compose([
    transforms.Resize((225, 225), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop((225, 225)),
    _convert_image_to_rgb,
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

dataset = CustomDataset(images, tags, transform)
dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

model, _ = load_model("ViT-B/16", "cuda")
logits = []
start = time.time()
for i_batch, sample_batched in enumerate(dataloader):
    logits_img = search_frames(sample_batched, caption, model)
    for logit in logits_img:
        logits.append(logit)

logits = np.array(logits)
logits = logits.reshape(-1,)
print(f"Time executed: {time.time() - start}")
ids = ((-logits).argsort()[:n_matches]).tolist()

#precision = window_precision(df, caption, ids, images)
#print(f"The precision of the tag captioning is: {precision:.4f}")

visualise_top_images(images[ids], logits[ids])


