import numpy as np
from .structure_data import *
import time

path = "/home/efs/datasets/BDD/bdd100k/labels/det_20/det_train.json"

def window_accuracy(path, caption, ids, imgs):
    df = concatenate_tags(path)
    caption = caption.lower()
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    imgs_window = imgs[ids]
    for row in df.iterrows():
        row = row[1]
        for img in imgs:
            if img != row.iloc[0]:
                continue
            else:
                vals = row.iloc[1:].values
                tags = retrieve_tags(vals)

                if caption in tags and img in imgs_window:
                    tp += 1
                elif caption in tags and img not in imgs_window:
                    fn += 1
                elif caption not in tags and img in imgs_window:
                    fp += 1
                else:
                    tn += 1

    return (tp + tn)/(tp + tn + fp + fn)




