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


def to_db(image: ee.Image):
    return image.log10().multiply(10)

def un_log(image: ee.Image):
    return ee.Image(10).pow(image.divide(10))

# Perform Despeckling Using Refined Lee Filter
# Adapted by Guido Lemoine as coded in the SNAP 3.0 S1TBX
def speckle_reduction(image: ee.Image):
    weights3 = ee.List.repeat(ee.List.repeat(1,3),3)
    kernel3 = ee.Kernel.fixed(3,3, weights3, 1, 1, False)

    mean3 = image.reduceNeighborhood(ee.Reducer.mean(), kernel3)
    variance3 = image.reduceNeighborhood(ee.Reducer.variance(), kernel3)

    # Use a sample of the 3x3 windows inside a 7x7 windows to determine gradients and directions
    sample_weights = ee.List([[0,0,0,0,0,0,0], [0,1,0,1,0,1,0],[0,0,0,0,0,0,0], [0,1,0,1,0,1,0], [0,0,0,0,0,0,0],
                              [0,1,0,1,0,1,0],[0,0,0,0,0,0,0]])

    sample_kernel = ee.Kernel.fixed(7,7, sample_weights, 3,3, False)

    # Calculate mean and variance for the sampled windows and store as 9 bands
    sample_mean = mean3.neighborhoodToBands(sample_kernel)
    sample_var = variance3.neighborhoodToBands(sample_kernel)

    # Determine the 4 gradients for the sampled windows
    gradients = sample_mean.select(1).subtract(sample_mean.select(7)).abs()
    gradients = gradients.addBands(sample_mean.select(6).subtract(sample_mean.select(2)).abs())
    gradients = gradients.addBands(sample_mean.select(3).subtract(sample_mean.select(5)).abs())
    gradients = gradients.addBands(sample_mean.select(0).subtract(sample_mean.select(8)).abs())

    # And find the maximum gradient amongst gradient bands
    max_gradient = gradients.reduce(ee.Reducer.max())

    # Create a mask for band pixels that are the maximum gradient
    gradmask = gradients.eq(max_gradient)

    # duplicate gradmask bands: each gradient represents 2 directions
    gradmask = gradmask.addBands(gradmask)

    # Determine the 8 directions
    directions = sample_mean.select(1).subtract(sample_mean.select(4)).gt(sample_mean.select(4).
                                                                          subtract(sample_mean.select(7))).multiply(1)
    directions = directions.addBands(sample_mean.select(6).subtract(sample_mean.select(4)).
                                     gt(sample_mean.select(4).subtract(sample_mean.select(2))).multiply(2))
    directions = directions.addBands(sample_mean.select(3).subtract(sample_mean.select(4)).
                                     gt(sample_mean.select(4).subtract(sample_mean.select(5))).multiply(3))
    directions = directions.addBands(sample_mean.select(0).subtract(sample_mean.select(4)).
                                     gt(sample_mean.select(4).subtract(sample_mean.select(8))).multiply(4))
    # The next 4 are the not() of the previous 4
    directions = directions.addBands(directions.select(0).Not().multiply(5))
    directions = directions.addBands(directions.select(1).Not().multiply(6))
    directions = directions.addBands(directions.select(2).Not().multiply(7))
    directions = directions.addBands(directions.select(3).Not().multiply(8))

    # Mask all values that are not 1-8
    directions = directions.updateMask(gradmask)

    # "collapse" the stack into a singe band image (due to masking, each pixel has just one value (1-8) in
    # its directional band, and is otherwise masked)
    directions = directions.reduce(ee.Reducer.sum())

    sample_stats = sample_var.divide(sample_mean.multiply(sample_mean))

    # Calculate localNoiseVariance
    sigmaV = (sample_stats.toArray().arraySort().arraySlice(0,0,5).
              arrayReduce(ee.Reducer.mean(), [0]))

    # Set up the 7*7 kernels for directional statistics
    rect_weights = (ee.List.repeat(ee.List.repeat(0,7),3).
                    cat(ee.List.repeat(ee.List.repeat(1,7),4)))

    diag_weights = ee.List([[1,0,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,1,0,0,0,0],
                            [1,1,1,1,0,0,0], [1,1,1,1,1,0,0], [1,1,1,1,1,1,0], [1,1,1,1,1,1,1]])

    rect_kernel = ee.Kernel.fixed(7,7, rect_weights, 3, 3, False)
    diag_kernel = ee.Kernel.fixed(7,7, diag_weights, 3, 3, False)

    # Create stacks for mean and variance using the original kernels. Mask with relevant direction.
    dir_mean = image.reduceNeighborhood(ee.Reducer.mean(), rect_kernel).updateMask(directions.eq(1))
    dir_var = image.reduceNeighborhood(ee.Reducer.variance(), rect_kernel).updateMask(directions.eq(1))

    dir_mean = dir_mean.addBands(image.reduceNeighborhood(ee.Reducer.mean(), diag_kernel).updateMask(directions.eq(2)))
    dir_var = dir_var.addBands(image.reduceNeighborhood(ee.Reducer.variance(), diag_kernel).updateMask(directions.eq(2)))

    # and add the bands for rotated kernels
    for i in range(1, 6):
        dir_mean = dir_mean.addBands(image.reduceNeighborhood(ee.Reducer.mean(),
                                                              rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)))
        dir_var = dir_var.addBands(image.reduceNeighborhood(ee.Reducer.variance(),
                                                            rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)))
        dir_mean = dir_mean.addBands(image.reduceNeighborhood(ee.Reducer.mean(),
                                                              diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)))
        dir_var = dir_var.addBands(image.reduceNeighborhood(ee.Reducer.variance(),
                                                            diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)))


    # "collapse" the stack into a single band image (due to masking, each pixel has just one value in its directional
    # band, and is otherwise masked)
    dir_mean = dir_mean.reduce(ee.Reducer.sum())
    dir_var = dir_var.reduce(ee.Reducer.sum())

    # A finally generate the filtered value
    varX = dir_var.subtract(dir_mean.multiply(dir_mean).multiply(sigmaV)).divide(sigmaV.add(1.0))

    b = varX.divide(dir_var)

    result = dir_mean.add(b.multiply(image.subtract(dir_mean)))
    return result.arrayGet(0)


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
    return labels, chips

def visualization(image):
    plt.imshow(image, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    plt.show()

if __name__ == '__main__':
    # Test this script
    ts = filter_sentinel1(bb=[-122.49192573971985, 49.02676423765457, -122.32937691159485, 49.245995407997384],
                          start='2021-11-01', end='2021-11-05')
    print(ts[0][0])
    visualization(ts[0][0])