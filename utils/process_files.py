import os
import pandas as pd
import numpy as np
import math
import pyarrow.parquet as pq

from nuscenes.nuscenes import NuScenes

def concat_configured_parquets(config_path):
    df = False
    for root, dirs, files in os.walk(config_path, topdown=False):
        for name in files:
            parquet_path = os.path.join(root, name)
            data = pd.read_parquet(parquet_path)
            if df is False:
                df = data
            else:
                df = pd.concat([df, data])
    return df

def save_top_paths(img_paths):
    file = open("top-tokens.txt", "w")
    for img_path in img_paths:
        file.write(img_path + "\n")
    file.close()


def find_sample_tokens_from_paths(img_paths, sensor="CAM_FRONT"):
    nusc = NuScenes(version="v1.0-trainval", dataroot="/home/ubuntu/users/mateusz/data/nuscenes", verbose=False)
    samples = nusc.sample
    tokens = []
    
    for idx, img_path in enumerate(img_paths):
        img_path = "/".join(img_path.split("/")[-3:])
        img_paths[idx] = img_path
    
    for img_path in img_paths:
        for sample in samples:
            sensor_token = sample["data"][sensor]
            sample_data_token = nusc.get("sample_data", sensor_token)
            if img_path == sample_data_token["filename"]:
                tokens.append(sample_data_token["sample_token"])
    
    file = open("top-tokens.txt", "w")
    for token in tokens:
        file.write(token + "\n")
    file.close()
