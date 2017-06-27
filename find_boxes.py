from __future__ import division
from __future__ import print_function

import argparse
import astropy.io.fits as fits
import numpy as np

import matplotlib.pyplot as plt
from numina.array.display.ximshow import ximshow
from numina.array.display.pause_debugplot import pause_debugplot
from numina.drps import get_system_drps
from rsix.ximplotxy import ximplotxy
from rsix.cosinebell import cosinebell


def find_boxes(fitsfile, channels, previous_boxes, debugplot):

    # read the 2d image
    with fits.open(fitsfile) as hdulist:
        image2d = hdulist[0].data
    naxis2, naxis1 = image2d.shape

    if debugplot % 10 != 0:
        ximshow(image2d, show=True,
                title='initial twilight image', debugplot=debugplot)

    # extract cross section
    nc1 = channels[0]
    nc2 = channels[1]
    ycut = np.median(image2d[:, nc1:(nc2+1)], axis=1)
    xcut = np.arange(naxis2) + 1
    ximplotxy(xcut, ycut, debugplot=debugplot,
              xlabel='y axis', ylabel='number of counts',
              title=fitsfile + " [" + str(nc1) + "," + str(nc2) + "]")

    # initial manipulation
    ycut -= np.median(ycut)  # subtract median
    ycut /= np.max(ycut)  # normalise values
    ycut *= -1  # invert signal to convert minima in maxima
    mask = cosinebell(n=ycut.size, fraction=0.10)
    ycut *= mask
    ximplotxy(xcut, ycut, debugplot=debugplot,
              xlabel='y axis', ylabel='reversed scale')

    # Fourier filtering
    xf = np.fft.fftfreq(xcut.size)
    yf = np.fft.fftpack.fft(ycut)
    ximplotxy(xf, yf.real, plottype='semilog',
              xlim=(0., 0.51), debugplot=debugplot)
    cut = (np.abs(xf) > 0.10)
    yf[cut] = 0.0
    ycut_filt = np.fft.ifft(yf).real
    ax = ximplotxy(xcut, ycut_filt, show=False,
                   xlabel='y axis', ylabel='reversed scale')
    refined_boxes = np.zeros(previous_boxes.size)
    nsearch = 20
    for ibox, box in enumerate(previous_boxes):
        iargmax = ycut_filt[box - nsearch:box + nsearch + 1].argmax()
        refined_boxes[ibox] = xcut[iargmax + box - nsearch]
    ax.vlines(previous_boxes, ymin=1.2, ymax=1.3, colors='magenta')
    ax.vlines(refined_boxes, ymin=1.4, ymax=1.5, colors='green')

    plt.show(block=False)
    plt.pause(0.001)
    pause_debugplot(debugplot)


def get_previous_boxes(vph, insmode, uuid):
    """Get previous boxes for the VPH, INSMODE and expected UUID.

    Using numina and megaradrp functionality.

    """

    d = get_system_drps()
    mydrp = d.drps['MEGARA']
    ic = mydrp.configurations[uuid]
    boxdict = ic.get('pseudoslit.boxes_positions',
                     **{'vph': vph, 'insmode': insmode})
    return np.array(boxdict['positions'])


def main(args=None):
    # parse command-line options
    parser = argparse.ArgumentParser(prog='find_boxes')
    # positional parameters
    parser.add_argument("fitsfile",
                        help="FITS image",
                        type=argparse.FileType('r'))
    parser.add_argument("--channels",
                        help="Channel region to extract cross section ",
                        default=(1990, 2010),
                        type=int, nargs=2)
    parser.add_argument("--debugplot",
                        help="integer indicating plotting/debugging" +
                             " (default=10)",
                        type=int, default=12,
                        choices=[0, 1, 2, 10, 11, 12, 21, 22])

    args = parser.parse_args(args=args)

    vph = 'LR-I'
    insmode = 'LCB'
    uuid = 'ca3558e3-e50d-4bbc-86bd-da50a0998a48'
    previous_boxes = get_previous_boxes(vph, insmode, uuid)
    print('Previous boxes:\n', previous_boxes)
    find_boxes(args.fitsfile.name, args.channels, previous_boxes,
               args.debugplot)


if __name__ == "__main__":

    main()
