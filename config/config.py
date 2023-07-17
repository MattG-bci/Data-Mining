import os
import pandas as pd
import numpy as np
import math
import pyarrow.parquet as pq


def configure_path(data_path):
    paths = []
    config_path = "./config/config_files"
    for root, dirs, files in os.walk(data_path, topdown=False):
        for name in files:
            path = os.path.join(root, name)
            paths.append(path)
    
    parquet_file = pd.DataFrame(data=paths, columns=["name"])
    parquet_name = data_path.split("/")[-1]
    if not os.path.exists(config_path):
        os.mkdir(config_path)
    parquet_file.to_parquet(f"./config/config_files/{parquet_name}.parquet", engine="fastparquet")

if __name__ == "__main__":
    n_paths = int(input("\nHow many directories with data would you like to configure?: "))
    for _ in range(n_paths):
        data_path = input("Please enter a full path to a directory here: ")
        configure_path(data_path)
