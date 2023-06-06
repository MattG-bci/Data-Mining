import argparse
import os
import numpy as np
from src.search_representation import clip_representations
from src.visualisiation import visualise_top_images

parser = argparse.ArgumentParser(description="Find the most matching frames to the caption...")
parser.add_argument("--caption", "-text")
parser.add_argument("--matches", "-number", type=int)
args = parser.parse_args()
caption = args.caption
n_matches = args.matches

path = "./utils/data/"
images = np.array([path + image for image in os.listdir(path)])
probs = clip_representations(images, caption)
ids = ((-probs).argsort()[:n_matches]).tolist()
visualise_top_images(images[ids], probs[ids])

