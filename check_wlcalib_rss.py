from __future__ import division
from __future__ import print_function

import argparse
import astropy.io.fits as fits
import numpy as np
import os

from numina.array.display.ximplot import ximplot
from numina.array.display.ximshow import ximshow
from numina.array.wavecalib.check_wlcalib import check_sp

from numina.array.display.pause_debugplot import DEBUGPLOT_CODES

from fix_borders_wlcalib_rss import fix_pix_borders


def process_rss(fitsfile, npix_zero_in_border,
                geometry, debugplot):
    """Process twilight image.

    Parameters
    ----------
    fitsfile : str
        Wavelength calibrated RSS FITS file name.
    npix_zero_in_border : int
        Number of pixels to be set to zero at the beginning and at
        the end of each spectrum to avoid unreliable pixel values
        produced in the wavelength calibration procedure.
    geometry : tuple (4 integers) or None
        x, y, dx, dy values employed to set the Qt backend geometry.
    debugplot : int
        Debugging level for messages and plots. For details see
        'numina.array.display.pause_debugplot.py'.

    Returns
    -------
    spmedian : numpy array
        Median spectrum corresponding to the collapse of the full
        RSS image.
    crpix1: float
        CRPIX1 keyword.
    crval1: float
        CRVAL1 keyword.
    cdelt1: float
        CDELT1 keyword.

    """

    # read the 2d image
    with fits.open(fitsfile) as hdulist:
        image2d_header = hdulist[0].header
        image2d = hdulist[0].data
    naxis2, naxis1 = image2d.shape
    crpix1 = image2d_header['crpix1']
    crval1 = image2d_header['crval1']
    cdelt1 = image2d_header['cdelt1']
    print('* Input file:', fitsfile)
    print('>>> NAXIS1:', naxis1)
    print('>>> NAXIS2:', naxis2)
    print('>>> CRPIX1:', crpix1)
    print('>>> CRVAL1:', crval1)
    print('>>> CDELT1:', cdelt1)

    if abs(debugplot) in (21, 22):
        ximshow(image2d, show=True,
                title='Wavelength calibrated RSS image', debugplot=debugplot)

    # set to zero a few pixels at the beginning and at the end of each
    # spectrum to avoid unreliable values coming from the wavelength
    # calibration procedure
    image2d = fix_pix_borders(image2d, nreplace=npix_zero_in_border,
                              sought_value=0, replacement_value=0)
    if abs(debugplot) in (21, 22):
        ximshow(image2d, show=True,
                title='RSS image after removing ' +
                      str(npix_zero_in_border) + ' pixels at the borders',
                debugplot=debugplot)

    # mask and masked array
    mask2d = (image2d == 0)
    image2d_masked = np.ma.masked_array(image2d, mask=mask2d)

    # median (and normalised) vertical cross section
    ycutmedian = np.ma.median(image2d_masked, axis=1).data
    # normalise cross section with its own median
    tmpmedian = np.median(ycutmedian)
    if tmpmedian > 0:
        ycutmedian /= tmpmedian
    else:
        raise ValueError('Unexpected null median in cross section')
    # replace zeros by ones
    iszero = np.where(ycutmedian == 0)
    ycutmedian[iszero] = 1
    if abs(debugplot) in (21, 22):
        ximplot(ycutmedian, plot_bbox=(1, naxis2),
                title='median ycut', debugplot=debugplot)

    # equalise the flux in each fiber by dividing the original image by the
    # normalised vertical cross secction
    ycutmedian2d = np.repeat(ycutmedian, naxis1).reshape(naxis2, naxis1)
    image2d_eq = image2d_masked/ycutmedian2d
    if abs(debugplot) in (21, 22):
        ximshow(image2d_eq.data, show=True,
                title='equalised image', debugplot=debugplot)

    # median spectrum
    spmedian = np.ma.median(image2d_eq, axis=0).data

    return spmedian, crpix1, crval1, cdelt1


def main(args=None):
    # parse command-line options
    parser = argparse.ArgumentParser(prog='check_wlcalib_rss')
    # positional parameters
    parser.add_argument("fitsfile",
                        help="Wavelength calibrated RSS FITS image",
                        type=argparse.FileType('r'))
    parser.add_argument("--wv_master_file", required=True,
                        help="TXT file containing wavelengths",
                        type=argparse.FileType('r'))
    parser.add_argument("--out_sp",
                        help="File name to save the median spectrum in FITS "
                             "format including the wavelength "
                             "calibration (default=None)",
                        default=None,
                        type=argparse.FileType('w'))
    parser.add_argument("--npixzero",
                        help="Number of pixels to be set to zero at the "
                             "borders of each spectrum (default=3)",
                        default=3, type=int)
    parser.add_argument("--geometry",
                        help="tuple x,y,dx,dy",
                        default="0,0,640,480")
    parser.add_argument("--debugplot",
                        help="integer indicating plotting/debugging" +
                             " (default=0)",
                        type=int, default=12,
                        choices=DEBUGPLOT_CODES)

    args = parser.parse_args(args=args)

    # geometry
    if args.geometry is None:
        geometry = None
    else:
        tmp_str = args.geometry.split(",")
        x_geom = int(tmp_str[0])
        y_geom = int(tmp_str[1])
        dx_geom = int(tmp_str[2])
        dy_geom = int(tmp_str[3])
        geometry = x_geom, y_geom, dx_geom, dy_geom

    # compute median spectrum and get wavelength calibration parameters
    spmedian, crpix1, crval1, cdelt1 = process_rss(
        args.fitsfile.name,
        args.npixzero,
        geometry=geometry,
        debugplot=args.debugplot
    )

    # save median spectrum
    if args.out_sp is not None:
        hdu = fits.PrimaryHDU(spmedian)
        hdu.header['CRPIX1'] = crpix1
        hdu.header['CRVAL1'] = crval1
        hdu.header['CDELT1'] = cdelt1
        hdu.writeto(args.out_sp, overwrite=True)

    # read list of expected arc lines
    master_table = np.genfromtxt(args.wv_master_file)
    wv_master = master_table[:, 0]
    if abs(args.debugplot) in (21, 22):
        print('wv_master:', wv_master)

    # check the wavelength calibration
    title = 'fitsfile: ' + os.path.basename(args.fitsfile.name) + \
            ' [collapsed median]\n' + \
            'wv_master: ' + os.path.basename(args.wv_master_file.name)
    check_sp(sp=spmedian,
             crpix1=crpix1,
             crval1=crval1,
             cdelt1=cdelt1,
             wv_master=wv_master,
             title=title,
             geometry=geometry,
             debugplot=args.debugplot)


if __name__ == "__main__":

    main()
