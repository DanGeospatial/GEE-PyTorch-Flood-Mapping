"""
Version: v1.2
Date: 2021-02-11
Authors: Mullissa A., Vollrath A., Braun, C., Slagter B., Balling J., Gou Y., Gorelick N.,  Reiche J.

MIT License

Copyright (c) 2021 Adugna Mullissa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files, to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import ee


# ---------------------------------------------------------------------------//
# Linear to db scale
# ---------------------------------------------------------------------------//

def lin_to_db(image):
    """
    Convert backscatter from linear to dB.

    Parameters
    ----------
    image : ee.Image
        Image to convert

    Returns
    -------
    ee.Image
        output image

    """
    bandNames = image.bandNames().remove('angle')
    db = ee.Image.constant(10).multiply(image.select(bandNames).log10()).rename(bandNames)
    return image.addBands(db, None, True)


def db_to_lin(image):
    """
    Convert backscatter from dB to linear.

    Parameters
    ----------
    image : ee.Image
        Image to convert

    Returns
    -------
    ee.Image
        output image

    """
    bandNames = image.bandNames().remove('angle')
    lin = ee.Image.constant(10).pow(image.select(bandNames).divide(10)).rename(bandNames)
    return image.addBands(lin, None, True)


def lin_to_db2(image):
    """
    Convert backscatter from linear to dB by removing the ratio band.

    Parameters
    ----------
    image : ee.Image
        Image to convert

    Returns
    -------
    ee.Image
        Converted image

    """
    db = ee.Image.constant(10).multiply(image.select(['VV', 'VH']).log10()).rename(['VV', 'VH'])
    return image.addBands(db, None, True)


# ---------------------------------------------------------------------------//
# Add ratio bands
# ---------------------------------------------------------------------------//

def add_ratio_lin(image):
    """
    Adding ratio band for visualization

    Parameters
    ----------
    image : ee.Image
        Image to use for creating band ratio

    Returns
    -------
    ee.Image
        Image containing the ratio band

    """
    ratio = image.addBands(image.select('VV').divide(image.select('VH')).rename('VVVH_ratio'))

    return ratio.set('system:time_start', image.get('system:time_start'))