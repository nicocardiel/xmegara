from __future__ import division
from __future__ import print_function

import argparse
import astropy.io.fits as fits
import numpy as np

from numina.array.display.ximplot import ximplot
from numina.array.display.ximshow import ximshow


# def ximplotxy(x, y, plottype=None,
#               xlim=None, ylim=None, debugplot=None):
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     if plottype == 'semilog':
#         ax.semilogy(x, y)
#     else:
#         ax.plot(x, y)
#
#     if xlim is not None:
#         ax.set_xlim(xlim)
#     if ylim is not None:
#         ax.set_ylim(ylim)
#
#     plt.show(block=False)
#     plt.pause(0.001)
#     pause_debugplot(debugplot)


def filtmask(sp, fmin=0.02, fmax=0.15, debugplot=0):
    """Filter spectrum in Fourier space and apply cosine bell.

    Parameters
    ----------
    sp : numpy array
        Spectrum to be filtered and masked.
    fmin : float
        Minimum frequency to be employed.
    fmax : float
        Maximum frequency to be employed.
    debugplot : int
        Determines whether intermediate computations and/or plots
        are displayed:
        00 : no debug, no plots
        01 : no debug, plots without pauses
        02 : no debug, plots with pauses
        10 : debug, no plots
        11 : debug, plots without pauses
        12 : debug, plots with pauses


    Returns
    -------
    sp_filtmask : numpy array
        Filtered and masked spectrum

    """

    # Fourier filtering
    xf = np.fft.fftfreq(sp.size)
    yf = np.fft.fft(sp)
    if debugplot % 10 != 0:
        ximplotxy(xf, yf.real, xlim=(0, 0.51),
                  plottype='semilog', debugplot=debugplot)

    cut = (np.abs(xf) > fmax)
    yf[cut] = 0.0
    cut = (np.abs(xf) < fmin)
    yf[cut] = 0.0
    if debugplot % 10 != 0:
        ximplotxy(xf, yf.real, xlim=(0, 0.51),
                  plottype='semilog', debugplot=debugplot)

    sp_filt = np.fft.ifft(yf).real
    if debugplot % 10 != 0:
        ximplot(sp_filt, title="filtered median spectrum",
                plot_bbox=(1, sp_filt.size), debugplot=debugplot)

    sp_filtmask = sp_filt * cosinebell(sp_filt.size, 0.1)
    if debugplot % 10 != 0:
        ximplot(sp_filtmask, title="filtered and masked median spectrum",
                plot_bbox=(1, sp_filt.size), debugplot=debugplot)

    return sp_filtmask


def periodic_corr1d(x, y):
    """Periodic correlation, implemented using FFT.

    x and y must be real sequences with the same length.

    Parameters
    ----------
    x : numpy array
        First sequence.
    y : numpy array
        Second sequence.

    Returns
    -------
    crosscorr : numpy array
        Periodic correlation

    """

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Invalid array dimensions")
    if x.shape != y.shape:
        raise ValueError("x and y shapes are different")

    corr = np.fft.ifft(np.fft.fft(x) * np.fft.fft(y).conj()).real

    return corr


def cosinebell(n, fraction):
    """Return a cosine bell spanning n pixels, masking a fraction of pixels

    Parameters
    ----------
    n : int
        Number of pixels.
    fraction : float
        Length fraction over which the data will be masked.

    """

    mask = np.ones(n)
    nmasked = int(fraction * n)
    for i in range(nmasked):
        yval = 0.5 * (1 - np.cos(np.pi * float(i) / float(nmasked)))
        mask[i] = yval
        mask[n - i - 1] = yval

    return mask


def process_twilight(fitsfile, debugplot):
    """Process twilight image.

    Parameters
    ----------
    fitsfile : str
        Twilight RSS FITS file name.
    debugplot : int
        Determines whether intermediate computations and/or plots
        are displayed:
        00 : no debug, no plots
        01 : no debug, plots without pauses
        02 : no debug, plots with pauses
        10 : debug, no plots
        11 : debug, plots without pauses
        12 : debug, plots with pauses

    """

    # read the 2d image
    with fits.open(fitsfile) as hdulist:
        image2d = hdulist[0].data
    if debugplot % 10 != 0:
        ximshow(image2d, show=True,
                title='initial twilight image', debugplot=debugplot)

    # compute median spectrum
    spmedian = np.median(image2d, axis=0)
    if debugplot % 10 != 0:
        ximplot(spmedian, title="median spectrum",
                plot_bbox=(1, spmedian.size), debugplot=debugplot)

    # compute filtered and masked median spectrum
    spmedian_filtmask = filtmask(spmedian, debugplot=debugplot)

    for i in range(5):
        sp_filtmask = filtmask(image2d[i,:])
        if debugplot % 10 != 0:
            ximplot(sp_filtmask, title="median spectrum of scan " + str(i),
                    plot_bbox=(1, spmedian.size), debugplot=debugplot)
        corr = periodic_corr1d(spmedian_filtmask, sp_filtmask)
        if debugplot % 10 != 0:
            ximplot(corr, title="periodic correlation with scan " + str(i),
                    plot_bbox=(1, corr.size), debugplot=debugplot)


def main(args=None):
    # parse command-line options
    parser = argparse.ArgumentParser(prog='twilight')
    # positional parameters
    parser.add_argument("fitsfile",
                        help="Twilight FITS image",
                        type=argparse.FileType('r'))
    parser.add_argument("--debugplot",
                        help="integer indicating plotting/debugging" +
                             " (default=10)",
                        type=int, default=12,
                        choices=[0, 1, 2, 10, 11, 12, 21, 22])

    args = parser.parse_args(args=args)

    process_twilight(args.fitsfile.name, args.debugplot)

if __name__ == "__main__":

    main()
