import numpy as np
import pandas as pd
import os

img_path = "/mnt/d/SAR_testing/images/"

def create_df(path):
    file_name = []
    for dirname, _, filenames in os.walk(path): # given a directory iterates over the files
        for filename in filenames:
            f = filename.split('.')[0]
            f = f.replace('tile_', '')
            file_name.append(f)

    return pd.DataFrame({'id': file_name}, index = np.arange(0, len(file_name))).sort_values('id').reset_index(drop=True)


x = create_df(img_path)['id'].values

print(x)