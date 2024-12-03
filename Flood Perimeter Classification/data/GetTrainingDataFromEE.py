"""
Get SAR images from Earth Engine to train network for three sites

"""
import ee
import geemap

# Initialize Earth Engine Api
# More information at developers.google.com/earth-engine/guides/python_install-conda#windows
ee.Initialize(project='ee-nelson-remote-sensing')

# Mask image edges
def mask_edge(image):
    edge = image.lt(-30.0)
    masked_image = image.mask().And(edge.Not())
    return image.updateMask(masked_image)

def to_db(image: ee.Image):
    return image.log10().multiply(10)

def un_log(image: ee.Image):
    return ee.Image(10).pow(image.divide(10))

# Perform Despeckling Using Refined Lee Filter
# Based on doi: 10.1109/IHMSC.2015.236
def speckle_reduction(image: ee.Image):

    pass

def export_numpy(image: ee.Image, aoi: ee.Geometry.BBox):
    sar_np = geemap.ee_to_numpy(image, region=aoi)
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
        .map(mask_edge)
        .mean()
    )

    filter_vh = (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.eq('resolution_meters', 10))
        .filter(ee.Filter.date(start, end))
        .select('VH')
        .map(mask_edge)
        .mean()
    )

    img_vv = filter_vv.clip(geom_box).map(un_log).map(speckle_reduction).map(to_db)
    img_vh = filter_vh.clip(geom_box).map(un_log).map(speckle_reduction).map(to_db)

    # Create squared multiplication (VH2*VV2) SAR index
    # Based on https://doi.org/10.1016/j.jag.2022.103002
    sar_cat = ee.Image.cat([img_vv, img_vh])
    sq_mul_sar = sar_cat.expression(
        '(VH ** 2) * (VV ** 2)', {
            'VH': sar_cat.select(['VH']),
            'VV': sar_cat.select(['VV'])
        }
    )

    return export_numpy(image=sq_mul_sar, aoi=geom_box)

