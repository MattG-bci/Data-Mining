import json
import pandas as pd
import numpy as np
import math

path = "/home/efs/datasets/BDD/bdd100k/labels/det_20/det_train.json"


def extract_objects(df):
    object_tags = []
    for row in df.iterrows():
        labels = row[1][-1]
        if labels != labels: # checking for nan
            continue
        objects = list(set([i["category"] for i in row[1][-1]]))
        object_tags.append(objects)
        
    return object_tags


def retrieve_tags(row):
    return list(filter(lambda x: x is not None, row))


def concatenate_tags(data_path):
    df = pd.read_json(data_path)
    scene_tags = [j[1][1] for j in df.iterrows()]
    img_names = [j[1][0] for j in df.iterrows()]
    object_tags = extract_objects(df)

    scene_tags = pd.DataFrame(scene_tags)
    img_names = pd.DataFrame(img_names)
    img_names.rename(columns={0 : "name"}, inplace=True)
    object_tags = pd.DataFrame(object_tags)
    concated = pd.concat([img_names, scene_tags, object_tags], axis=1)
    return concated


if __name__ == "__main__":
    df = concatenate_tags(path)
    row = df.iloc[0, 1:].values
    print(retrieve_tags(row))