import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image


class RGBDataset(Dataset):
    def __init__(self, img_path, mask_path, X, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform

    def __len__(self):
        # Get total number of samples
        return len(self.X)

    def __getitem__(self, index):

        image = Image.open(self.img_path + self.X[index] + '.png').convert("RGB")
        image = np.array(image, dtype=np.uint8)

        assert image.max() <= 255.0 and image.min() >= 0

        if self.transform is not None:
            aug = self.transform(image=image)
            image = aug['image']

        image = ToTensor()(image)

        msk = Image.open(self.mask_path + self.X[index] + '.png').convert('L')
        label = np.array(msk, dtype=np.uint8)
        label[label > 254] = 1
        assert label.max() <= 1.0 and label.min() >= 0

        label = torch.tensor(label, dtype=torch.long)

        return image, label
