import numpy as np
import albumentations as ab
from torch.utils.data import Dataset
from PIL import Image

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

        image = np.array(Image.open(self.img_path + 'tile_' + self.X[index] + '.png'))
        mask = np.array(Image.open(self.mask_path + 'tile_' + self.X[index] + '.png'))

        # Scale image values to 0-255 to make it easier for working with albumentations
        image = ((image - image.min()) * (1/(image.max() - image.min()) * 255)).astype('uint8')

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']

        normalized = ab.Normalize()(image=image, mask=np.expand_dims(mask, 0))
        # normalized = ab.Normalize()(image=np.expand_dims(image, 2), mask=np.expand_dims(mask, 0)) for greyscale

        assert mask.max() <= 1.0 and mask.min() >= 0

        return normalized["image"].transpose(2, 0, 1), normalized["mask"]