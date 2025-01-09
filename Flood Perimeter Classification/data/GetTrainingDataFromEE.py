"""
Get SAR images from Earth Engine to train network for three sites

"""
import ee
import geemap
import numpy
import SAR_Processing.wrapper as wp
import multiprocessing
import os
import requests
import shutil
from retry import retry


# Initialize Earth Engine Api
# More information at developers.google.com/earth-engine/guides/python_install-conda#windows
ee.Initialize(project='ee-nelson-remote-sensing', url="https://earthengine-highvolume.googleapis.com")


def getRequests(image: ee.Image, params: dict, region: ee.Geometry):
    img = ee.Image(1).rename("Class").addBands(image)
    points = img.stratifiedSample(
        numPoints=params["count"],
        region=region,
        scale=params["scale"],
        seed=params["seed"],
        geometries=True,
    )

    return points.aggregate_array(".geo").getInfo()

@retry(tries=10, delay=1, backoff=2)
def getResult(index, point, image: ee.Image, params: dict, mask: ee.Image):
    point = ee.Geometry.Point(point["coordinates"])
    region = point.buffer(params["buffer"]).bounds()

    if params["format"] in ["png", "jpg"]:
        url = image.getThumbURL(
            {
                "region": region,
                "dimensions": params["dimensions"],
                "format": params["format"],
            }
        )
        url_m = mask.getThumbURL(
            {
                "region": region,
                "dimensions": params["dimensions"],
                "format": params["format"],
            }
        )
    else:
        url = image.getDownloadURL(
             {
                "region": region,
                "dimensions": params["dimensions"],
                "format": params["format"],
             }
         )
        url_m = mask.getDownloadURL(
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

    r_m = requests.get(url_m, stream=True)
    if r_m.status_code != 200:
        r_m.raise_for_status()

    out_dir = os.path.abspath(params["out_dir"])
    basename = str(index).zfill(len(str(params["count"])))
    filename_images = f"{out_dir}/images/{params['prefix']}{basename}.{ext}"
    filename_masks = f"{out_dir}/masks/{params['prefix']}{basename}.{ext}"
    with open(filename_images, "wb") as out_file:
        shutil.copyfileobj(r_img.raw, out_file)
        print("Done: ", basename)

    with open(filename_masks, "wb") as out_file:
        shutil.copyfileobj(r_m.raw, out_file)
        print("Done: ", basename)


def filter_sentinel1(mask: ee.Image, start: str, end: str):

    # Retrieve SAR data under mask
    geomimg = mask.geometry()

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

    params = {
        "count": 100,  # How many image chips to export
        "buffer": 127,  # The buffer distance (m) around each point
        "scale": 100,  # The scale to do stratified sampling
        "seed": 32,  # A randomization seed to use for subsampling.
        "dimensions": "256x256",  # The dimension of each image chip
        "format": "NPY",  # The output image format, can be png, jpg, ZIPPED_GEO_TIFF, GEO_TIFF, NPY
        "prefix": "tile_",  # The filename prefix
        "processes": 20,  # How many processes to used for parallel processing
        "out_dir": "/mnt/d/SAR_testing",  # The output directory. Default to the current working directly
    }

    items = getRequests(image=sq_mul_sar, params=params, region=geomimg)

    pool = multiprocessing.Pool(params["processes"])
    pool.starmap(getResult(image=sq_mul_sar, params=params, mask=mask), enumerate(items))

    pool.close()


# Test this script
if __name__ == '__main__':

    # Load in labels
    label = ee.Image("projects/ee-nelson-remote-sensing/assets/SARMask")

    filter_sentinel1(mask=label,
                          start='2021-11-17', end='2021-11-22')
