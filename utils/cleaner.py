import numpy as np
import os
from PIL import Image

img_path = "/mnt/d/SAR_Cat/images/"
mask_path = "/mnt/d/SAR_Cat/masks/"

with os.scandir(img_path) as it:
    for file in it:
        msk = Image.open(mask_path + file.name).convert("RGB")
        red, green, blue = msk.split()

        extrema = green.getextrema()
        if extrema == (0, 0):
            os.remove(path=(img_path + file.name))
            os.remove(path=(mask_path + file.name))
        else:
            continue


