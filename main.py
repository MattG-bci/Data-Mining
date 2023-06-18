import argparse
import os
import numpy as np
import pandas as pd
from src.search_representation import search_frames, load_model
from src.visualisiation import visualise_top_images
from src.metrics import window_precision
from utils.DataLoader import *
from utils.from_configured_files import import_configured_parquets
from src.structure_data import *
from torch.utils.data import DataLoader
from torchvision import transforms
import warnings
import time

warnings.filterwarnings("ignore")

caption = input("\nWhat is your keyword?: ")
n_matches = int(input("How many top matches would you like to see?: "))

def process_caption(caption):
    if "_" in caption:
        caption = caption.replace("_", " ")
    return caption

caption = process_caption(caption.lower())

#config_path = "./config/config_files"
#df = import_configured_parquets(config_path)

#### Original lines for the BDD data ###
train_path = "./src/pq_labels/det_train_new.parquet"
val_path = "./src/pq_labels/det_val_new.parquet"

df_train = concatenate_tags(train_path)
df_val = concatenate_tags(val_path)
df = pd.concat([df_train, df_val])
### END ###

n_samples = int(input(f"How many samples do you wish to consider (max: {len(df)})?: "))
print("\n################ PROCESSING - PLEASE WAIT ################")

df = df.iloc[0:n_samples, :]
images = df.name.values
#tags = df.iloc[:, 1:].values

def _convert_image_to_rgb(image):
    return image.convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop((224, 224)),
    _convert_image_to_rgb,
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

dataset = CustomDataset(images, transform)
dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

model, _ = load_model("ViT-B/16", "cuda")
logits = np.array([])
start = time.time()
for i_batch, sample_batched in enumerate(dataloader):
    logits_img = search_frames(sample_batched, caption, model)
    logits = np.concatenate((logits, logits_img), axis=None)


print(f"Time executed: {(time.time() - start):.4f} seconds")
ids = ((-logits).argsort()[:n_matches]).tolist()

#### To test the accuracy of the model ####
#precision = window_precision(df, caption, ids, images)
#print(f"The precision of the tag captioning is: {precision:.4f}")
#### END ####

visualise_top_images(images[ids], logits[ids])


