import torch
import numpy
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt

img_pth = "/mnt/d/SAR_testing/masks/tile_1.NPY"

def transform_numpy(npy):
    # Scale image values to 0-255 to make it easier for working with torch
    transformed_np = ((npy - npy.min()) * (1/(npy.max() - npy.min()) * 255)).astype('uint8')
    return transformed_np

def visualization(img):
    plt.imshow(img, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    plt.show()


if __name__ == '__main__':
    image = numpy.load(img_pth).astype(dtype=numpy.float64)
    print(numpy.info(image))
    trans = transform_numpy(image)
    visualization(trans)