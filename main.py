import argparse
import os
import numpy as np
from src.search_representation import search_frames
from src.visualisiation import visualise_top_images
from src.metrics import window_accuracy

parser = argparse.ArgumentParser(description="Find the most matching frames to the caption...")
parser.add_argument("--caption", "-text")
parser.add_argument("--matches", "-number", type=int)
args = parser.parse_args()
caption = args.caption
n_matches = args.matches

def process_caption(caption):
    if "_" in caption:
        caption = caption.replace("_", " ")
    return caption

caption = process_caption(caption)

path = "./utils/data/"
images = np.array([path + image for image in os.listdir(path)])
probs = search_frames(images, caption)
ids = ((-probs).argsort()[:n_matches]).tolist()

tagged_imgs = np.array([image for image in os.listdir(path)])
tag_path = "/home/efs/datasets/BDD/bdd100k/labels/det_20/det_train.json"
acc = window_accuracy(tag_path, caption, ids, tagged_imgs)
print(f"The accuracy of the tag captioning is: {acc:.4f}")

visualise_top_images(images[ids], probs[ids])


