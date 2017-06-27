from __future__ import division
from __future__ import print_function

import argparse
import astropy.io.fits as fits

from numina.array.display.ximshow import ximshow


def process_twilight(fits_file, traces_file):
    """Process twilight image.

    Parameters
    ----------
    fits_file : str
        Twilight FITS file name.
    traces_file : str
        JSON file with fiber traces.

    """

    # read the new image
    with fits.open(fits_file) as hdulist:
        image2d = hdulist[0].data

    ximshow(image2d, show=True)


def main(args=None):
    # parse command-line options
    parser = argparse.ArgumentParser(prog='twilight')
    # positional parameters
    parser.add_argument("fits_file",
                        help="Twilight FITS image",
                        type=argparse.FileType('r'))
    parser.add_argument("traces_file",
                        help="JSON file with fiber traces",
                        type=argparse.FileType('r'))

    args = parser.parse_args(args=args)

    process_twilight(args.fits_file.name, args.traces_file.name)

if __name__ == "__main__":

    main()
