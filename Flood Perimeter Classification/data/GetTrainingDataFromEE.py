"""
Get SAR images from Earth Engine to train network for three sites

"""
import ee
import geemap
import numpy
import matplotlib.pyplot as plt
import SAR_Processing.wrapper as wp


# Initialize Earth Engine Api
# More information at developers.google.com/earth-engine/guides/python_install-conda#windows
ee.Initialize(project='ee-nelson-remote-sensing')


def export_numpy(image: ee.Image, aoi: ee.Geometry.BBox):
    # Convert image to numpy
    sar_np = geemap.ee_to_numpy(image, region=aoi, scale=10)
    # Scale image values to 0-255 to make it easier for working with torch
    transformed_np = ((sar_np - sar_np.min()) * (1/(sar_np.max() - sar_np.min()) * 255)).astype('uint8')
    return transformed_np

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

    # Set empty Lists
    labels = []
    chips = []

    # Create grid of 224 chips
    grid = geemap.create_grid(mask, 224)

    # Split the labels into 224 chips
    for cell in range(grid.size().getInfo()):
        gd = grid.select(cell)
        cp = mask.clip(gd)
        # Export labels into list of NumPy
        ex = export_numpy(image=cp, aoi=gd)
        labels.append(ex)

    # Split the chips into 224 chips
    for cell in range(grid.size().getInfo()):
        gd = grid.select(cell)
        cp = sq_mul_sar.clip(gd)
        # Export chips into list of NumPy
        ex = export_numpy(image=cp, aoi=gd)
        chips.append(ex)

    # Return List of Lists
    return [labels, chips]

def visualization(image):
    plt.imshow(image, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    plt.show()

# Test this script
if __name__ == '__main__':

    # Load in labels
    label = ee.Image("projects/ee-nelson-remote-sensing/assets/SARMask")

    ts = filter_sentinel1(mask=label,
                          start='2021-11-17', end='2021-11-22')
    print(ts[0][0])
    visualization(ts[0][0])
    visualization(ts[1][0])