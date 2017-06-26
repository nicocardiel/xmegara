from __future__ import division
from __future__ import print_function

import argparse
import json
import numpy as np
from numpy.polynomial import Polynomial

from numina.array.display.ximshow import ximshow_file
from numina.array.display.pause_debugplot import pause_debugplot


def main(args=None):
    # parse command-line options
    parser = argparse.ArgumentParser(prog='overplot_traces')
    # positional parameters
    parser.add_argument("fits_file",
                        help="FITS image containing the spectra",
                        type=argparse.FileType('r'))
    parser.add_argument("traces_file",
                        help="JSON file with traces",
                        type=argparse.FileType('r'))
    # optional parameters
    parser.add_argument("--rawimage",
                        help="FITS file is a RAW image (otherwise RSS assumed)",
                        action="store_true")
    parser.add_argument("--yoffset",
                        help="Vertical offset (+upwards, -downwards)",
                        default=0,
                        type=float)
    parser.add_argument("--fibids",
                        help="Display fiber identification number",
                        action="store_true")
    parser.add_argument("--z1z2",
                        help="tuple z1,z2, minmax or None (use zscale)")
    parser.add_argument("--bbox",
                        help="bounding box tuple: nc1,nc2,ns1,ns2")
    parser.add_argument("--keystitle",
                        help="tuple of FITS keywords.format: " +
                             "key1,key2,...keyn.'format'")
    parser.add_argument("--geometry",
                        help="tuple x,y,dx,dy")

    args = parser.parse_args(args=args)

    ax = ximshow_file(args.fits_file.name,
                      args_z1z2=args.z1z2,
                      args_bbox=args.bbox,
                      args_keystitle=args.keystitle,
                      args_geometry=args.geometry,
                      show=False)

    # trace offsets for RAW images
    if args.rawimage:
        ix_offset = 51
    else:
        ix_offset = 1

    # read traces from JSON file
    bigdict = json.loads(open(args.traces_file.name).read())
    for fiberdict in bigdict['contents']:
        xmin = fiberdict['start']
        xmax = fiberdict['stop']
        fibid = fiberdict['fibid']
        coeff = fiberdict['fitparms']
        # skip fibers without trace
        if len(coeff) > 0:
            num = int(float(xmax-xmin+1)+0.5)
            xp = np.linspace(start=xmin, stop=xmax, num=num)
            ypol = Polynomial(coeff)
            yp = ypol(xp)
            if args.rawimage:
                lcut = (yp > 2056.5)
                yp[lcut] += 100
            yp += args.yoffset
            ax.plot(xp+ix_offset, yp, 'b:')
            if args.fibids:
                ax.text((xmin+xmax)/2, yp[int(num/2)], str(fibid), fontsize=6,
                        color='green', backgroundcolor='white')
        else:
            print('>>> Missing fiber:', fibid)

    import matplotlib.pyplot as plt
    plt.show(block=False)
    plt.pause(0.001)
    pause_debugplot(12)

if __name__ == "__main__":

    main()
