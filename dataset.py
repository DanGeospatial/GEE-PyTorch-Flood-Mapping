"""

Copyright (C) 2025 Daniel Nelson

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

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
