from __future__ import division
from __future__ import print_function

import argparse
import json
import numpy as np
from numpy.polynomial import Polynomial


def traces2ds9(json_file, ds9_file, rawimage, numpix=100, fibid_at=0):
    """Transfor fiber traces from JSON to ds9-region format.

    Parameters
    ----------
    json_file : str
        Input JSON file name.
    ds9_file : file
        Handle to output file name in ds9-region format.
    rawimage : bool
        If True the traces must be generated to be overplotted on
        raw FITS images.
    numpix : int
        Number of abscissae per fiber trace.
    fibid_at : int
        Abscissae where the fibid is shown (default=0 -> not shown).

    """

    # offset between polynomial and image abscissae
    if rawimage:
        ix_offset = 51
    else:
        ix_offset = 1

    # open output file and insert header

    ds9_file.write('# Region file format: DS9 version 4.1\n')
    ds9_file.write('global color=green dashlist=2 4 width=1 '
                   'font="helvetica 10 normal roman" select=1 '
                   'highlite=1 dash=1 fixed=0 edit=1 '
                   'move=1 delete=1 include=1 source=1\n')
    ds9_file.write('physical\n')

    # read traces from JSON file and save region in ds9 file
    bigdict = json.loads(open(json_file).read())
    for fiberdict in bigdict['contents']:
        fibid = fiberdict['fibid']
        xmin = fiberdict['start']
        xmax = fiberdict['stop']
        coeff = np.array(fiberdict['fitparms'])
        # skip fibers without trace
        ds9_file.write('# fibid: ' + str(fibid) + '\n')
        if len(coeff) > 0:
            xp = np.linspace(start=xmin, stop=xmax, num=numpix)
            ypol = Polynomial(coeff)
            yp = ypol(xp)
            for i in range(len(xp)-1):
                x1 = xp[i] + ix_offset
                y1 = yp[i] + 1
                x2 = xp[i+1] + ix_offset
                y2 = yp[i+1] + 1
                ds9_file.write('line ' + str(x1) + ' ' + str(y1) + ' ' +
                               str(x2) + ' ' + str(y2) + '\n')
                if fibid_at != 0:
                    if x1 <= fibid_at <= x2:
                        ds9_file.write('text ' + str((x1+x2)/2) + ' ' +
                                       str((y1+y2)/2) + ' {' + str(fibid) +
                                       '}  # color=blue\n')
        else:
            print('Warning ---> Missing fiber:', fibid)


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
    parser.add_argument("--numpix",
                        help="Number of pixels/trace (default 100)",
                        default=100, type=int)
    parser.add_argument("--rawimage",
                        help="FITS file is a RAW image (RSS assumed instead)",
                        action="store_true")
    parser.add_argument("--fibid_at",
                        help="Display fiber identification number at location",
                        default=0, type=int)

    args = parser.parse_args(args=args)

    traces2ds9(args.json_file.name, args.ds9_file, args.rawimage,
               args.numpix, args.fibid_at)


if __name__ == "__main__":

    main()
