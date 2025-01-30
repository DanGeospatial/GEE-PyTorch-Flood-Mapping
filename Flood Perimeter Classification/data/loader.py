import numpy as np
import pandas as pd
import os
import albumentations as ab
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from .dataset import NPYDataset

img_path = "/mnt/d/SAR_Cat/images/"
mask_path = "/mnt/d/SAR_Cat/masks/"

train_ratio = 0.60
validation_ratio = 0.20
test_ratio = 0.20

loader_args = dict(num_workers=os.cpu_count(), pin_memory=True)

def create_df(path):
    file_name = []
    for dirname, _, filenames in os.walk(path): # given a directory iterates over the files
        for filename in filenames:
            f = filename.split('.')[0]
            f = f.replace('tile_', '')
            file_name.append(f)

    return pd.DataFrame({'id': file_name}, index = np.arange(0, len(file_name))).sort_values('id').reset_index(drop=True)


x = create_df(img_path)['id'].values

x_train, x_test = train_test_split(x, test_size=1 - train_ratio, random_state=32)

# test is now 20% of the initial data set
# validation is now 20% of the initial data set
x_val, x_test = train_test_split(x_test,
                                                test_size=test_ratio / (test_ratio + validation_ratio),
                                                random_state=32)

print(f'Train Size: {len(x_train)}')
print(f'Val Size: {len(x_val)}')
print(f'Test Size: {len(x_test)}')

tran = ab.Compose([ab.HorizontalFlip(p=0.5), ab.VerticalFlip(p=0.5), ab.GaussNoise(p=0.2),
               ab.GridDistortion(p=0.2), ab.RandomBrightnessContrast((0,0.5),(0,0.5), p=0.2)])

# create train dataset with augmentations and then valid and test without
train_dataset = NPYDataset(img_path, mask_path, x_train, tran)
valid_dataset = NPYDataset(img_path, mask_path, x_val)
test_dataset = NPYDataset(img_path, mask_path, x_test)

# load images in batch size dependent on VRAM
batch_size = 3

# Make sure to shuffle train but NOT valid or test
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_args)
val_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, **loader_args)
test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_args)
