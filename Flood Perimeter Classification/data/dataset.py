import numpy as np
import albumentations as ab
from torch.utils.data import Dataset


class NPYDataset(Dataset):
    def __init__(self, img_path, mask_path, X, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform

    def __len__(self):
        # Get total number of samples
        return len(self.X)

    def __getitem__(self, index):

        image = np.array(np.load(file=(self.img_path + self.X[index] + '.NPY').astype(dtype=np.float64), allow_pickle=True))
        mask = np.array(np.load(file=(self.mask_path + self.X[index] + '.NPY').astype('uint8'), allow_pickle=True))

        # Scale image values to 0-255 to make it easier for working with albumentations
        image = ((image - image.min()) * (1/(image.max() - image.min()) * 255)).astype('uint8')

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']

        normalized = ab.Normalize()(image=image, mask=np.expand_dims(mask, 0))

        assert mask.max() <= 1.0 and mask.min() >= 0

        return normalized["image"].transpose(2, 0, 1), normalized["mask"]