"""
Get SAR images from Earth Engine to train network for three sites

"""
import ee
import geemap
import numpy
import matplotlib.pyplot as plt

# Initialize Earth Engine Api
# More information at developers.google.com/earth-engine/guides/python_install-conda#windows
ee.Initialize(project='ee-nelson-remote-sensing')


def export_numpy(image: ee.Image, aoi: ee.Geometry.BBox):
    # Convert image to numpy
    sar_np = geemap.ee_to_numpy(image, region=aoi, scale=10)
    # Scale image values to 0-255 to make it easier for working with torch
    transformed_np = ((sar_np - sar_np.min()) * (1/(sar_np.max() - sar_np.min()) * 255)).astype('uint8')
    return transformed_np

def filter_sentinel1(bb: list, start: str, end: str):

    geom_box = ee.Geometry.BBox(west=bb[0], south=bb[1], east=bb[2], north=bb[3])

    filter_vv = (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.eq('resolution_meters', 10))
        .filter(ee.Filter.date(start, end))
        .select('VV')
        .mean()
    )

    filter_vh = (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.eq('resolution_meters', 10))
        .filter(ee.Filter.date(start, end))
        .select('VH')
        .mean()
    )

    img_vv = filter_vv.clip(geom_box)
    #img_vv_un_log = un_log(img_vv)
    #img_vv_sp_reduction = speckle_reduction(img_vv_un_log)
    #img_vv_to_db = to_db(img_vv_sp_reduction).rename('VV')

    img_vh = filter_vh.clip(geom_box)
    #img_vh_un_log = un_log(img_vh)
    #img_vh_sp_reduction = speckle_reduction(img_vh_un_log)
    #img_vh_to_db = to_db(img_vh_sp_reduction).rename('VH')

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
    # Make sure to clip these both to same extent first

    # Load in labels
    label = ee.Image("users/danielnelsonca/Research/SARMask")
    # Create grid of 224 chips
    grid = geemap.create_grid(label, 224)

    # Split the labels into 224 chips
    for cell in range(grid.size().getInfo()):
        gd = grid.select(cell)
        cp = label.clip(gd)
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
    ts = filter_sentinel1(bb=[-122.49192573971985, 49.02676423765457, -122.32937691159485, 49.245995407997384],
                          start='2021-11-01', end='2021-11-05')
    print(ts[0][0])
    visualization(ts[0][0])