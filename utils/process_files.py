import os
import pandas as pd
import numpy as np
import math
import pyarrow.parquet as pq


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
            
