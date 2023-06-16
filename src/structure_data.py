import json
import pandas as pd
import numpy as np
import math
import pyarrow.parquet as pq
import os


def extract_objects(df):
    object_tags = []
    for row in df.iterrows():
        labels = row[1][-1]
        if labels is None: # checking for None
            continue
        objects = list(set([i["category"] for i in row[1][-1]]))
        object_tags.append(objects)
        
    return object_tags


def retrieve_tags(row):
    return list(filter(lambda x: x is not None and x is not float("nan"), row))


def concatenate_tags(data_path):
    df = pd.read_parquet(data_path, engine="fastparquet")
    scene_tags = [j[1][1] for j in df.iterrows()]
    img_names = [j[1][0] for j in df.iterrows()]
    object_tags = extract_objects(df)

    scene_tags = pd.DataFrame(scene_tags)
    img_names = pd.DataFrame(img_names)
    img_names.rename(columns={0 : "name"}, inplace=True)
    object_tags = pd.DataFrame(object_tags)
    concated = pd.concat([img_names, scene_tags, object_tags], axis=1)
    concated = concated.where(pd.notnull(concated), None)
    return concated


def save_json_to_parquet(path, img_path, set_name):
    df = pd.read_json(path)
    img_names = [j[1][0] for j in df.iterrows()]
    ids = []
    for root, dirs, files in os.walk(img_path, topdown=False):
        for name in files:
            for idx, img_name in enumerate(img_names):
                if img_name == name:
                    point = idx 
                    img_names[idx] = os.path.join(root, name)
                    df.iloc[idx, 0] = os.path.join(root, name)
                    ids.append(idx)

            
    df = df[df.index.isin(ids)]   
    df.to_parquet(f"pq_labels/det_{set_name}_new.parquet", engine="fastparquet")


if __name__ == "__main__":
    path = "./pq_labels/det_train_new.parquet"
    df = concatenate_tags(path)
    print(df)
    #save_json_to_parquet("/home/efs/datasets/BDD/bdd100k/labels/det_20/det_val.json", "/home/efs/datasets/BDD/bdd100k/images/10k/train", "val")
