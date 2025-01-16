import numpy as np
import os

img_path = "/mnt/d/SAR_Cat/images/"
mask_path = "/mnt/d/SAR_Cat/masks/"

with os.scandir(img_path) as it:
    for file in it:
        image = np.array(np.load(file=(img_path + file.name), allow_pickle=True).astype(dtype=np.float64))
        mask = np.array(np.load(file=(mask_path + file.name), allow_pickle=True).astype('uint8'))

        if np.isin(mask.any(), 1):
            break
        else:
            os.remove(path=(img_path + file.name))
            os.remove(path=(mask_path + file.name))

