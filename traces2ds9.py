from __future__ import division
from __future__ import print_function

import argparse
import json
import numpy as np
from numpy.polynomial import Polynomial


def traces2ds9(json_file, ds9_file, rawimage, xstep=10):
    """Transfor fiber traces from JSON to ds9-region format.

    Parameters
    ----------
    json_file : str
        Input JSON file name.
    ds9_file : str
        Output file name in ds9-region format.
    rawimage : bool
        If True the traces must be generated to be overplotted on
        raw FITS images.
    xstep : int
        Abscissa step to compute polynomial.

    """

    # offset between polynomial and image abscissae
    if rawimage:
        ix_offset = 51
    else:
        ix_offset = 1

    # open output file and insert header
    f = open(ds9_file, 'w')
    f.write('# Region file format: DS9 version 4.1')
    f.write('global color=green dashlist=8 3 width=1 font="helvetica 10 '
            '"normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 '
            'move=1 delete=1 include=1 source=1')
    f.write('physical')

    # read traces from JSON file and save region in ds9 file
    bigdict = json.loads(open(json_file).read())
    for fiberdict in bigdict['contents']:
        fibid = fiberdict['fibid']
        xmin = fiberdict['start']
        xmax = fiberdict['stop']
        coeff = np.array(fiberdict['fitparms'])
        # skip fibers without trace
        if len(coeff) > 0:
            xp = np.arange(xmin, xmax + 1, xstep)
            ypol = Polynomial(coeff)
            yp = ypol(xp)
            for i in range(len(xp)-1):
                x1 = str(xp[i] + ix_offset)
                y1 = str(yp[i])
                x2 = str(xp[i+1] + ix_offset)
                y2 = str(yp[i+1])
                f.write('line ' + x1 + ' ' + y1 + ' ' + x2 + ' ' + y2)
        else:
            print('Warning ---> Missing fiber:', fibid)

    # close output file
    f.close()


def main(args=None):
    # parse command-line options
    parser = argparse.ArgumentParser(prog='overplot_traces')
    # positional parameters
    parser.add_argument("json_file",
                        help="JSON file with fiber traces",
                        type=argparse.FileType('r'))
    parser.add_argument("ds9_file",
                        help="Output region file in ds9 format",
                        type=argparse.FileType('w'))
    # optional parameters
    parser.add_argument("--rawimage",
                        help="FITS file is a RAW image (RSS assumed instead)",
                        action="store_true")
    parser.add_argument("--fibids",
                        help="Display fiber identification number",
                        action="store_true")

    args = parser.parse_args(args=args)

    traces2ds9(args.json_file.name, args.ds9_file.name, args.rawimage)


if __name__ == "__main__":

    main()
