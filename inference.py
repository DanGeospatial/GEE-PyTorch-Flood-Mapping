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

import os
import numpy as np
import torch
import ee
import geemap
from PIL import Image
from torchgeo.models import FarSeg
import matplotlib.pyplot as plt
import rasterio
from rasterio import MemoryFile, merge
from rasterio.plot import show
from torch import device, cuda, autocast
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset, stack_samples
from torchgeo.samplers import GridGeoSampler


Image.MAX_IMAGE_PIXELS = None
image_path = '/mnt/d/water/SAR_BC.tif'
output_path = '/mnt/d/water/test_output_bc.tif'
loader_args = dict(num_workers=os.cpu_count(), pin_memory=False)
batch_size = 10
num_classes = 2
chip_stride = 512
size = 1024


print("Using PyTorch version: ", torch.__version__)
device = device('cuda' if cuda.is_available() else 'cpu')
print(f"Inferencing on {device}")
# Load the model
model = FarSeg(backbone='resnet18', classes=num_classes, backbone_pretrained=True).to(device)
model.load_state_dict(torch.load("/mnt/d/SARFloodModel.pth"))
model.eval()

input_drone_image = RasterDataset(image_path)
inference_sampler = GridGeoSampler(input_drone_image, size=size, stride=chip_stride)
inference_set = DataLoader(input_drone_image , batch_size=batch_size, sampler=inference_sampler, collate_fn=stack_samples,
                       **loader_args)


bounds_list = []
image_list = []
crs_list = []

with (torch.inference_mode()):
    for batch in inference_set:
        images = batch["image"]
        bs = images.shape[0]
        images = images.to(device=device)

        with autocast(device.type):
            output = model(images.half())


        for i in range(bs):
            bb = batch["bounds"][i]
            im = output[i].cpu()
            cr = batch["crs"][i]
            bounds_list.append(bb)
            image_list.append(im)
            crs_list.append(cr)


raster_list = []

for i in range(len(image_list)):
    trans = rasterio.transform.from_bounds(west=float(bounds_list[i][0]), north=float(bounds_list[i][3]),
                                           east=float(bounds_list[i][1]), south=float(bounds_list[i][2]), width=size,
                                           height=size)

    pred = torch.argmax(image_list[i], dim=0).numpy().astype(np.uint8)

    profile = {
        'driver': 'GTiff',
        'height': size,
        'width': size,
        'count': 1,  # Number of bands,
        'dtype': np.uint8,
        'crs': crs_list[i],
        'transform': trans
    }
    memfile = MemoryFile()
    rst = memfile.open(**profile)
    rst.write(pred, 1)
    raster_list.append(rst)


mosaic, out_trans = merge.merge(raster_list, method='min')
# show(mosaic, cmap='tab10')
# Update the metadata
out_meta = {"driver": "GTiff",
                 "height": mosaic.shape[1],
                 "width": mosaic.shape[2],
                 "transform": out_trans,
                 "crs": crs_list[0],
                 'dtype': np.uint8,
                 'count': 1,
                 }
with rasterio.open(output_path, "w", **out_meta) as dest:
    dest.write(mosaic)

ras = raster_list[1].read()
show(ras, cmap='tab10')
# Convert to segmentation map
segmentation_map = torch.argmax(image_list[2], dim=0).numpy().astype(np.uint8)
# Visualize the segmentation map
plt.imshow(segmentation_map, cmap='tab10', interpolation='none')  # Use a colormap that supports 5 classes (0-4)
# plt.title("Reassembled Segmentation Map")
# plt.colorbar()
# plt.axis('off')
plt.show()
