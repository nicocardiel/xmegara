from __future__ import division
from __future__ import print_function

import argparse
import astropy.io.fits as fits
import numpy as np

from scipy import fftpack
from numina.array.display.ximshow import ximshow


def find_boxes(fitsfile, debugplot):
    # read the 2d image
    with fits.open(fitsfile) as hdulist:
        image2d = hdulist[0].data
    if debugplot % 10 != 0:
        ximshow(image2d, show=True,
                title='initial twilight image', debugplot=debugplot)


def main(args=None):
    # parse command-line options
    parser = argparse.ArgumentParser(prog='find_boxes')
    # positional parameters
    parser.add_argument("fitsfile",
                        help="FITS image",
                        type=argparse.FileType('r'))
    parser.add_argument("--debugplot",
                        help="integer indicating plotting/debugging" +
                             " (default=10)",
                        type=int, default=12,
                        choices=[0, 1, 2, 10, 11, 12, 21, 22])

    args = parser.parse_args(args=args)

    find_boxes(args.fitsfile.name, args.debugplot)

if __name__ == "__main__":

    main()
