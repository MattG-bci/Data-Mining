import argparse
import os
import numpy as np
import pandas as pd
from src.search_representation import search_frames, search_model
from src.visualisiation import visualise_top_images
from src.metrics import window_precision
from utils.DataLoader import *
from src.structure_data import *
from torch.utils.data import DataLoader
from torchvision import transforms
import warnings
import time

parser = argparse.ArgumentParser(description="Find the most matching frames to the caption...")
parser.add_argument("--caption", "-text")
parser.add_argument("--matches", "-number", type=int)
args = parser.parse_args()
caption = args.caption
n_matches = args.matches

warnings.filterwarnings("ignore")

def process_caption(caption):
    if "_" in caption:
        caption = caption.replace("_", " ")
    return caption

caption = process_caption(caption)

train_path = "./src/pq_labels/det_train_new.parquet"
val_path = "./src/pq_labels/det_val_new.parquet"

df_train = concatenate_tags(train_path)
df_val = concatenate_tags(val_path)
df = pd.concat([df_train, df_val])
images = df.name.values
tags = df.iloc[:, 1:].values
transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop((224, 224))
])

dataset = CustomDataset(images, tags, transform)
dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

model, _ = search_model("ViT-B/16", "cuda")
start = time.time()
for i_batch, sample_batched in enumerate(dataloader):
    probs = search_frames(sample_batched, caption, model)
print(f"Time executed: {time.time() - start}")
#ids = ((-probs).argsort()[:n_matches]).tolist()


#precision = window_precision([train_path, val_path], caption, ids, images)
#print(f"The precision of the tag captioning is: {precision:.4f}")

#visualise_top_images(images[ids], probs[ids])


