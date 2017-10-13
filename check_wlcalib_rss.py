from __future__ import division
from __future__ import print_function

import argparse
import astropy.io.fits as fits
from fix_borders_wlcalib_rss import fix_pix_borders
import numpy as np
from numina.array.display.pause_debugplot import pause_debugplot
from numina.array.display.ximplot import ximplot
from numina.array.display.ximshow import ximshow
from numina.array.display.ximplotxy import ximplotxy

from numina.array.display.pause_debugplot import DEBUGPLOT_CODES


def process_rss(fitsfile, npix_zero_in_border, linelist, debugplot):
    """Process twilight image.

    Parameters
    ----------
    fitsfile : str
        Wavelength calibrated RSS FITS file name.
    npix_zero_in_border : int
        Number of pixels to be set to zero at the beginning and at
        the end of each spectrum to avoid unreliable pixel values
        produced in the wavelength calibration procedure.
    linelist : file handler
        ASCII file with the detailed list of expected arc lines.
    debugplot : int
        Debugging level for messages and plots. For details see
        'numina.array.display.pause_debugplot.py'.

    """

    # read the 2d image
    with fits.open(fitsfile) as hdulist:
        image2d_header = hdulist[0].header
        image2d = hdulist[0].data
    naxis2, naxis1 = image2d.shape
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
    if abs(debugplot) in (21, 22):
        xdum = np.arange(naxis1) + 1
        ax = ximplotxy(xdum, spmedian, show=False,
                       title="median spectrum", label='initial')
        ax.legend()
        pause_debugplot(debugplot, pltshow=True)


def main(args=None):
    # parse command-line options
    parser = argparse.ArgumentParser(prog='twilight')
    # positional parameters
    parser.add_argument("fitsfile",
                        help="Wavelength calibrated RSS FITS image",
                        type=argparse.FileType('r'))
    parser.add_argument("linelist",
                        help="ASCII file with detailed list of expected "
                             "arc lines",
                        type=argparse.FileType('r'))
    parser.add_argument("--npixzero",
                        help="Number of pixels to be set to zero at the "
                             "borders of each spectrum (default=3)",
                        default=3, type=int)
    parser.add_argument("--debugplot",
                        help="integer indicating plotting/debugging" +
                             " (default=0)",
                        type=int, default=0,
                        choices=DEBUGPLOT_CODES)

    args = parser.parse_args(args=args)

    process_rss(args.fitsfile.name,
                args.npixzero,
                args.linelist,
                args.debugplot)


if __name__ == "__main__":

    main()
