"""
Get SAR images from Earth Engine to image chips

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
import ee
import geemap
import numpy
import SAR_Helpers.wrapper as wp
import multiprocessing
import os
import requests
import shutil
from retry import retry


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
        url_m = mask.getThumbURL(
            {
                'min': 0, 'max': 1,
                'bands': ["b1"],
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
        "count": 3000,  # How many image chips to export
        "buffer": 227,  # The buffer distance (m) around each point
        "scale": 100,  # The scale to do stratified sampling
        "seed": 32,  # A randomization seed to use for subsampling.
        "dimensions": "1024x1024",  # The dimension of each image chip
        "format": "png",  # The output image format, can be png, jpg, ZIPPED_GEO_TIFF, GEO_TIFF, NPY
        "prefix": "tile_",  # The filename prefix
        "processes": 5,  # How many processes to used for parallel processing
        "out_dir": "/mnt/d/SAR_Cat",  # The output directory. Default to the current working directly
    }

    items = getRequests(image=mask, params=params, region=geomimg)

    pool = multiprocessing.Pool(params["processes"])
    pool.starmap(getResult, enumerate(items))

    pool.close()


def inference_sentinel1(lbl: ee.FeatureCollection, start: str, end: str, out_name: str):

    # Retrieve SAR data under mask
    geomimg = lbl.geometry()

    parameter_vv = {'START_DATE': start,
                 'STOP_DATE': end,
                 'POLARIZATION': 'VV',
                 'ORBIT': 'ASCENDING',
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
                 'ORBIT': 'ASCENDING',
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

    vv_min = img_vv.select(['VV']).reduceRegion(reducer= ee.Reducer.min(), geometry=geomimg, scale=100)
    vv_max= img_vv.select(['VV']).reduceRegion(reducer= ee.Reducer.max(), geometry=geomimg, scale=100)
    vv_norm = img_vv.select(['VV']).unitScale(low=vv_min.get('VV'), high=vv_max.get('VV'))

    vh_min = img_vh.select(['VH']).reduceRegion(reducer=ee.Reducer.min(), geometry=geomimg, scale=100)
    vh_max = img_vh.select(['VH']).reduceRegion(reducer=ee.Reducer.max(), geometry=geomimg, scale=100)
    vh_norm = img_vh.select(['VH']).unitScale(low=vh_min.get('VH'), high=vh_max.get('VH'))

    avg_min = sq_avg.select(['VH']).reduceRegion(reducer= ee.Reducer.min(), geometry=geomimg, scale=100)
    avg_max = sq_avg.select(['VH']).reduceRegion(reducer= ee.Reducer.max(), geometry=geomimg, scale=100)
    avg_norm = sq_avg.select(['VH']).unitScale(low=avg_min.get('VH'), high=avg_max.get('VH'))

    img_cat = ee.Image.cat([vv_norm, vh_norm, avg_norm])

    # These images are larger than 32-48 MB, so I will need to use this if not splitting them
    task = ee.batch.Export.image.toDrive(
        image=img_cat,
        description=out_name,
        region=lbl.geometry(),
        crs='EPSG:3857',
        scale=1,
        maxPixels=300000000
    )
    task.start()


# Test this script
if __name__ == '__main__':

    get_tiles = False
    get_BC = False
    get_ethiopia = False
    get_Oakville = False

    if get_tiles:
        # Load in labels
        label = ee.Image("projects/ee-nelson-remote-sensing/assets/BC_Water")
        filter_sentinel1(lbl=label,
                          start='2020-07-01', end='2020-08-25')

    if get_BC:
        label = ee.FeatureCollection('projects/ee-nelson-remote-sensing/assets/BCFlooding')
        inference_sentinel1(lbl=label, start='2020-07-01', end='2020-08-25', out_name='SAR_BC')

    if get_ethiopia:
        label = ee.FeatureCollection('projects/ee-nelson-remote-sensing/assets/EthiopiaDam')
        inference_sentinel1(lbl=label, start='2021-07-01', end='2021-08-25', out_name='SAR_eth')

    if get_Oakville:
        label = ee.FeatureCollection('projects/ee-nelson-remote-sensing/assets/OakvilleFlooding')
        inference_sentinel1(lbl=label, start='2024-07-01', end='2024-07-28', out_name='SAR_OAK')

