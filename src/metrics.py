import numpy as np
from .structure_data import *
import time


def window_precision(paths, caption, ids, imgs):
    df_train = concatenate_tags(paths[0])
    df_val = concatenate_tags(paths[1])

    df = pd.concat([df_train, df_val])
    caption = caption.lower()
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    imgs_window = imgs[ids]
    for img in imgs:
        row = df[df["name"] == img]
        vals = row.iloc[:, 1:].values
        if vals.any():
            vals = vals[0]
            tags = retrieve_tags(vals)

            if caption in tags and img in imgs_window:
                tp += 1
            elif caption in tags and img not in imgs_window:
                fn += 1
            elif caption not in tags and img in imgs_window:
                fp += 1
            else:
                tn += 1
    
    return tp/(tp + fp)




