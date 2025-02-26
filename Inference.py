import torch
from torch import device, cuda
import segmentation_models_pytorch as smp
import numpy as np
import albumentations as ab
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

import data.SAR_Processing.wrapper as wp
import multiprocessing
import ee
from retry import retry
import requests
import shutil

img_path = "/mnt/d/SAR_Test/images/"

loader_args = dict(num_workers=os.cpu_count(), pin_memory=True)

# Initialize Earth Engine Api
# More information at developers.google.com/earth-engine/guides/python_install-conda#windows
ee.Initialize(project='ee-nelson-remote-sensing', url="https://earthengine-highvolume.googleapis.com")


def getRequests(image: ee.Image, params: dict, region: ee.Geometry):
    imger = ee.Image(1).rename("Class").addBands(image)
    points = imger.stratifiedSample(
        numPoints=params["count"],
        region=region,
        scale=params["scale"],
        seed=params["seed"],
        geometries=True,
    )

    return points.aggregate_array(".geo").getInfo()

@retry(tries=10, delay=1, backoff=2)
def getResult(index, point):
    point = ee.Geometry.Point(point["coordinates"])
    region = point.buffer(params["buffer"]).bounds()

    if params["format"] in ["png", "jpg"]:
        url = img.getThumbURL(
            {
                'min': -30, 'max': 1,
                "region": region,
                "dimensions": params["dimensions"],
                "format": params["format"],
            }
        )
    else:
        url = img.getDownloadURL(
            {
                "region": region,
                "dimensions": params["dimensions"],
                "format": params["format"],
            }
        )

    if params["format"] == "GEO_TIFF":
        ext = "tif"
    else:
        ext = params["format"]

    r_img = requests.get(url, stream=True)
    if r_img.status_code != 200:
        r_img.raise_for_status()

    out_dir = os.path.abspath(params["out_dir"])
    basename = str(index).zfill(len(str(params["count"])))
    filename_images = f"{out_dir}/images/{params['prefix']}{basename}.{ext}"
    with open(filename_images, "wb") as out_file:
        shutil.copyfileobj(r_img.raw, out_file)
        print("Done: ", basename)


def filter_sentinel1(lbl: ee.Image, start: str, end: str):

    # Retrieve SAR data under mask
    global mask
    mask = lbl
    geomimg = lbl.geometry()

    parameter_vv = {'START_DATE': start,
                 'STOP_DATE': end,
                 'POLARIZATION': 'VV',
                 'ORBIT': 'DESCENDING',
                 'ROI': geomimg,
                 'APPLY_BORDER_NOISE_CORRECTION': False,
                 'APPLY_SPECKLE_FILTERING': True,
                 'SPECKLE_FILTER_FRAMEWORK': 'MULTI',
                 'SPECKLE_FILTER': 'GAMMA MAP',
                 'SPECKLE_FILTER_KERNEL_SIZE': 9,
                 'SPECKLE_FILTER_NR_OF_IMAGES': 10,
                 'APPLY_TERRAIN_FLATTENING': True,
                 'DEM': ee.Image('USGS/SRTMGL1_003'),
                 'TERRAIN_FLATTENING_MODEL': 'VOLUME',
                 'TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER': 0,
                 'FORMAT': 'DB',
                 'CLIP_TO_ROI': False,
                 'SAVE_ASSET': False,
                 'ASSET_ID': "users/XXX"
                 }

    parameter_vh = {'START_DATE': start,
                 'STOP_DATE': end,
                 'POLARIZATION': 'VH',
                 'ORBIT': 'DESCENDING',
                 'ROI': geomimg,
                 'APPLY_BORDER_NOISE_CORRECTION': False,
                 'APPLY_SPECKLE_FILTERING': True,
                 'SPECKLE_FILTER_FRAMEWORK': 'MULTI',
                 'SPECKLE_FILTER': 'GAMMA MAP',
                 'SPECKLE_FILTER_KERNEL_SIZE': 9,
                 'SPECKLE_FILTER_NR_OF_IMAGES': 10,
                 'APPLY_TERRAIN_FLATTENING': True,
                 'DEM': ee.Image('USGS/SRTMGL1_003'),
                 'TERRAIN_FLATTENING_MODEL': 'VOLUME',
                 'TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER': 0,
                 'FORMAT': 'DB',
                 'CLIP_TO_ROI': False,
                 'SAVE_ASSET': False,
                 'ASSET_ID': "users/XXX"
                 }


    filter_vv = wp.s1_preproc(parameter_vv)
    filter_vh = wp.s1_preproc(parameter_vh)

    # Mean then Clip to mask geometry
    img_vv = filter_vv.mean().clip(geomimg)
    img_vh = filter_vh.mean().clip(geomimg)

    # Create squared multiplication (VH2*VV2) SAR index
    # Based on https://doi.org/10.1016/j.jag.2022.103002
    sar_cat = ee.Image.cat([img_vv, img_vh])
    sq_mul_sar = sar_cat.expression(
        '(VH ** 2) * (VV ** 2)', {
            'VH': sar_cat.select(['VH']),
            'VV': sar_cat.select(['VV'])
        }
    )
    sq_avg = sar_cat.expression(
        '(VH + VV) / 2', {
            'VH': sar_cat.select(['VH']),
            'VV': sar_cat.select(['VV'])
        }
    )

    img_cat = ee.Image.cat([sar_cat.select(['VV']), sar_cat.select(['VH']), sq_avg.select(['VH'])])

    # Doing this is not very good
    global params
    global img
    img = img_cat

    params = {
        "count": 3,  # How many image chips to export
        "buffer": 227,  # The buffer distance (m) around each point
        "scale": 100,  # The scale to do stratified sampling
        "seed": 32,  # A randomization seed to use for subsampling.
        "dimensions": "512x512",  # The dimension of each image chip
        "format": "png",  # The output image format, can be png, jpg, ZIPPED_GEO_TIFF, GEO_TIFF, NPY
        "prefix": "tile_",  # The filename prefix
        "processes": 20,  # How many processes to used for parallel processing
        "out_dir": "/mnt/d/SAR_Test",  # The output directory. Default to the current working directly
    }

    items = getRequests(image=mask, params=params, region=geomimg)

    pool = multiprocessing.Pool(params["processes"])
    pool.starmap(getResult, enumerate(items))

    pool.close()

def create_df(path):
    file_name = []
    for dirname, _, filenames in os.walk(path): # given a directory iterates over the files
        for filename in filenames:
            f = filename.split('.')[0]
            f = f.replace('tile_', '')
            file_name.append(f)

    return pd.DataFrame({'id': file_name}, index = np.arange(0, len(file_name))).sort_values('id').reset_index(drop=True)

class NPYDataset(Dataset):
    def __init__(self, img_path_class, X):
        self.img_path = img_path_class
        self.X = X

    def __len__(self):
        # Get total number of samples
        return len(self.X)

    def __getitem__(self, index):

        image = np.array(Image.open(self.img_path + 'tile_' + self.X[index] + '.png').convert("RGB"))

        normalized = ab.Normalize()(image=image)

        return normalized["image"].transpose(2, 0, 1)



def visualization(vis_tp: tuple):
    image = vis_tp[0][0].cpu()
    mask = vis_tp[1][0].cpu()

    plt.figure(figsize=(14, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    plt.show()

    image = vis_tp[0][1].cpu()
    mask = vis_tp[1][1].cpu()

    plt.figure(figsize=(14, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    plt.show()


def inference_model(model, device_hw, dl):

    tiles = []
    pred = []

    for img in dl:
        model.eval()
        with torch.inference_mode():
            images = img.to(device=device_hw)
            mask_prediction = model(images)

            tiles.append(img)
            pred.append(mask_prediction)

    return tiles, pred


if __name__ == '__main__':
    print("Using PyTorch version: ", torch.__version__)

    device = device('cuda' if cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    # Load in labels
    label = ee.Image("projects/ee-nelson-remote-sensing/assets/SARMask")

    # filter_sentinel1(lbl=label,
    #                      start='2021-11-16', end='2021-11-22')

    x = create_df(img_path)['id'].values
    print(f'Inference Size: {len(x)}')

    inference_dataset = NPYDataset(img_path, x)

    # load images in batch size dependent on VRAM
    batch_size = 1

    val_dl = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False, **loader_args)

    classes = 1

    model = smp.UnetPlusPlus(encoder_name="efficientnet-b2", in_channels=3, classes=classes, encoder_weights="imagenet").to(device)
    model.load_state_dict(torch.load("/mnt/d/SAR_Water_v1.pth", weights_only=True, map_location=device))

    visualization(inference_model(model=model, device_hw=device, dl=val_dl))
