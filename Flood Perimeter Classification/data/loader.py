import torch
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt



def transform_numpy(npy):
    # Scale image values to 0-255 to make it easier for working with torch
    transformed_np = ((npy - npy.min()) * (1/(npy.max() - npy.min()) * 255)).astype('uint8')
    return transformed_np

def visualization(image):
    plt.imshow(image, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    plt.show()