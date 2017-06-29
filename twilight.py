from __future__ import division
from __future__ import print_function

import argparse
import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyval
from numina.array.display.pause_debugplot import pause_debugplot
from numina.array.display.ximplot import ximplot
from numina.array.display.ximshow import ximshow
from numina.array.display.ximplotxy import ximplotxy
from numina.array.interpolation import SteffenInterpolator
from numina.array.wavecalib.peaks_spectrum import refine_peaks_spectrum


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
    if debugplot in (21, 22):
        ximplotxy(xf, yf.real, xlim=(0, 0.51),
                  plottype='semilog', debugplot=debugplot)

    cut = (np.abs(xf) > fmax)
    yf[cut] = 0.0
    cut = (np.abs(xf) < fmin)
    yf[cut] = 0.0
    if debugplot in (21, 22):
        ximplotxy(xf, yf.real, xlim=(0, 0.51),
                  plottype='semilog', debugplot=debugplot)

    sp_filt = np.fft.ifft(yf).real
    if debugplot in (21, 22):
        ximplot(sp_filt, title="filtered median spectrum",
                plot_bbox=(1, sp_filt.size), debugplot=debugplot)

    sp_filtmask = sp_filt * cosinebell(sp_filt.size, 0.1)
    if debugplot in (21, 22):
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


def oversample1d(sp, crval1, cdelt1, oversampling=1):
    """Oversample spectrum.

    Parameters
    ----------
    sp : numpy array
        Spectrum to be oversampled.
    crval1 : float
        Abscissae of the center of the first pixel in the original
        spectrum 'sp'.
    cdelt1 : float
        Abscissae increment corresponding to 1 pixel in the original
        spectrum 'sp'.
    oversampling : int
        Oversampling value per pixel.

    Returns
    -------
    sp_over : numpy array
        Oversampled data array.
    crval1_over : float
        Abscissae of the center of the first pixel in the oversampled
        spectrum.
    cdelt1_over : float
        Abscissae of the center of the last pixel in the oversampled
        spectrum.

    """

    if sp.ndim != 1:
        raise ValueError('Unexpected array dimensions')

    naxis1 = sp.size
    naxis1_over = naxis1 * oversampling
    cdelt1_over = cdelt1 / oversampling
    xmin = crval1 - cdelt1/2   # left border of first pixel
    crval1_over = xmin + cdelt1_over / 2

    sp_over = np.zeros(naxis1_over)
    for i in range(naxis1):
        i1 = i * oversampling
        i2 = i1 + oversampling
        sp_over[i1:i2] = sp[i]

    # crvaln = crval1 + (naxis1 - 1) * cdelt1
    # crvaln_over = crval1_over + (naxis1_over - 1) * cdelt1_over
    # xover = np.linspace(crval1_over, crvaln_over, naxis1_over)
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(np.linspace(crval1, crvaln, naxis1), sp, 'bo')
    # ax.plot(xover, sp_over, 'r+')
    # plt.show()

    return sp_over, crval1_over, cdelt1_over


def process_twilight(fitsfile, oversampling, debugplot):
    """Process twilight image.

    Parameters
    ----------
    fitsfile : str
        Twilight RSS FITS file name.
    oversampling : int
        Oversampling of each pixel.
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
    naxis2, naxis1 = image2d.shape
    if debugplot % 10 != 0:
        ximshow(image2d, show=True,
                title='initial twilight image', debugplot=debugplot)

    # median spectrum
    spmedian = np.median(image2d, axis=0)
    if debugplot % 10 != 0:
        ximplot(spmedian, title="median spectrum",
                plot_bbox=(1, naxis1), debugplot=debugplot)

    # filtered and masked median spectrum
    spmedian_filtmask = filtmask(spmedian, debugplot=debugplot)

    # periodic correlation

    xcorr = np.arange(naxis1)
    naxis1_half = int(naxis1/2)
    for i in range(naxis1_half):
        xcorr[i + naxis1_half] -= naxis1
    isort = xcorr.argsort()
    xcorr = xcorr[isort]
    naxis2_half = int(naxis2/2)

    offsets = np.zeros(naxis2)
    for i in range(naxis2):
        sp_filtmask = filtmask(image2d[i, :])
        if i == naxis2_half and (debugplot in (21, 22)):
            ximplot(sp_filtmask, title="median spectrum of scan " + str(i),
                    plot_bbox=(1, spmedian.size), debugplot=debugplot)
        corr = periodic_corr1d(sp_filtmask, spmedian_filtmask)
        corr = corr[isort]
        if i == naxis2_half and (debugplot in (21, 22)):
            ximplotxy(xcorr, corr,
                      title="periodic correlation with scan " + str(i),
                      xlim=(-20, 20), debugplot=debugplot)
        ixpeak = np.array([corr.argmax()])
        xdum, sdum = refine_peaks_spectrum(corr, ixpeak, 7,
                                           method='gaussian')
        offsets[i] = xdum - naxis1_half

    ximplotxy(np.arange(naxis2)+1, offsets, ylim=(-10, 10),
              xlabel='pixel in the NAXIS2 direction',
              ylabel='offset (pixels) in the NAXIS1 direction',
              debugplot=debugplot)

    # oversampling
    naxis1_over = naxis1 * oversampling
    image2d_over = np.zeros((naxis2, naxis1_over))
    xplot = np.linspace(1, naxis1, naxis1)
    xplot_over = np.linspace(1, naxis1, naxis1_over)
    for i in range(naxis2):
        sp_over, crval1_over, cdelt1_over = \
            oversample1d(image2d[i, :], crval1=1, cdelt1=1,
                         oversampling=oversampling)
        sp_over_shifted = shiftx_image2d_flux(sp_over,
                                              -offsets[i]*oversampling)
        image2d_over[i] = sp_over_shifted
        # ax = ximplotxy(xplot, spmedian, title='scan' + str(i), show=False)
        # ax.plot(xplot_over, sp_over, 'b+')
        # ax.plot(xplot_over, sp_over_shifted, 'g.')
        # ax.set_xlim([1760, 1790])
        # plt.show(block=False)
        # plt.pause(0.001)
        # pause_debugplot(debugplot)
        # plt.close()

    ximshow(image2d_over, debugplot=debugplot)

    # spmedian_over = np.zeros(naxis1*oversampling)
    # for i in range(naxis2):
    #     sp_filtmask = filtmask(image2d[i, :])
    #
    #
    # ximplotxy(np.arange(naxis1*nover)+1, sp_oversampled,
    #           debugplot=debugplot)


def shiftx_image2d_flux(image2d_orig, xoffset):
    """Resample a 2D image using a shift in the x direction.

    Parameters
    ----------
    image2d_orig : numpy array
        2D image to be resampled.
    xoffset : float
        Offset to be applied.

    Returns
    -------
    image2d_resampled : numpy array
        Resampled 2D image.

    """

    if image2d_orig.ndim == 1:
        naxis1 = image2d_orig.size
    elif image2d_orig.ndim ==2:
        naxis2, naxis1 = image2d_orig.shape
    else:
        print('>>> image2d_orig.shape:', image2d_orig.shape)
        raise ValueError('Unexpected number of dimensions')

    return resample_image2d_flux(image2d_orig,
                                 naxis1=naxis1,
                                 cdelt1=1,
                                 crval1=1,
                                 crpix1=1,
                                 coeff=[xoffset, 1])


def resample_image2d_flux(image2d_orig,
                          naxis1, cdelt1, crval1, crpix1, coeff):
    """Resample a 2d image using NAXIS1, CDELT1, CRVAL1, and CRPIX1.

    The same NAXIS1, CDELT1, CRVAL1, and CRPIX1 are employed for all
    the scans (rows) of the original 'image2d'. The wavelength
    calibrated output image has dimensions NAXIS1 * NSCAN, where NSCAN
    is the original number of scans (rows) of the original image.

    Parameters
    ----------
    image2d_orig : numpy array
        2D image to be resampled.
    naxis1 : int
        NAXIS1 of the resampled image.
    cdelt1 : float
        CDELT1 of the resampled image.
    crval1 : float
        CRVAL1 of the resampled image.
    crpix1 : float
        CRPIX1 of the resampled image.
    coeff : numpy array
        Coefficients of the wavelength calibration polynomial.

    Returns
    -------
    image2d_resampled : numpy array
        Wavelength calibrated 2D image
    """

    # duplicate input array, avoiding problems when using as input
    # 1d numpy arrays with shape (nchan,) instead of a 2d numpy
    # array with shape (1,nchan)
    if image2d_orig.ndim == 1:
        nscan = 1
        nchan = image2d_orig.size
        image2d = np.zeros((nscan, nchan))
        image2d[0, :] = np.copy(image2d_orig)
    elif image2d_orig.ndim ==2:
        nscan, nchan = image2d_orig.shape
        image2d = np.copy(image2d_orig)
    else:
        print('>>> image2d_orig.shape:', image2d_orig.shape)
        raise ValueError('Unexpected number of dimensions')

    new_x = np.arange(naxis1)
    new_wl = crval1 + cdelt1 * new_x

    old_x_borders = np.arange(-0.5, nchan)
    old_x_borders += crpix1  # following FITS criterium
    # old_wl_borders = polyval(old_x_borders, wcalib.T)

    new_borders = map_borders(new_wl)

    accum_flux = np.empty((nscan, nchan + 1))
    accum_flux[:, 1:] = np.cumsum(image2d, axis=1)
    accum_flux[:, 0] = 0.0
    image2d_resampled = np.zeros((nscan, naxis1))

    old_wl_borders = polyval(old_x_borders, coeff)

    for iscan in range(nscan):
        # We need a monotonic interpolator
        # linear would work, we use a cubic interpolator
        interpolator = SteffenInterpolator(
            old_wl_borders,
            accum_flux[iscan],
            extrapolate='border'
        )
        fl_borders = interpolator(new_borders)
        image2d_resampled[iscan] = fl_borders[1:] - fl_borders[:-1]

    if image2d_orig.ndim == 1:
        return image2d_resampled[0, :]
    else:
        return image2d_resampled


def map_borders(wls):
    """Compute borders of pixels for interpolation.

    The border of the pixel is assumed to be midway of the wls
    """
    midpt_wl = 0.5 * (wls[1:] + wls[:-1])
    all_borders = np.zeros((wls.shape[0] + 1,))
    all_borders[1:-1] = midpt_wl
    all_borders[0] = 2 * wls[0] - midpt_wl[0]
    all_borders[-1] = 2 * wls[-1] - midpt_wl[-1]
    return all_borders


def main(args=None):
    # parse command-line options
    parser = argparse.ArgumentParser(prog='twilight')
    # positional parameters
    parser.add_argument("fitsfile",
                        help="Twilight FITS image",
                        type=argparse.FileType('r'))
    parser.add_argument("oversampling",
                        help="Oversampling (1=none; default)",
                        default=1, type=int)
    parser.add_argument("--debugplot",
                        help="integer indicating plotting/debugging" +
                             " (default=10)",
                        type=int, default=12,
                        choices=[0, 1, 2, 10, 11, 12, 21, 22])

    args = parser.parse_args(args=args)

    process_twilight(args.fitsfile.name, args.oversampling, args.debugplot)


if __name__ == "__main__":

    main()
