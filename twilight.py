from __future__ import division
from __future__ import print_function

import argparse
import astropy.io.fits as fits
from fix_borders_wlcalib_rss import find_pix_borders
from fix_borders_wlcalib_rss import fix_pix_borders
import numpy as np
from numina.array.display.pause_debugplot import pause_debugplot
from numina.array.display.polfit_residuals import \
    polfit_residuals_with_sigma_rejection
from numina.array.display.ximplot import ximplot
from numina.array.display.ximshow import ximshow
from numina.array.display.ximplotxy import ximplotxy
from numina.array.wavecalib.peaks_spectrum import refine_peaks_spectrum
from numina.array.wavecalib.resample import oversample1d
from numina.array.wavecalib.resample import rebin
from numina.array.wavecalib.resample import shiftx_image2d_flux
from scipy import ndimage

from numina.array.display.pause_debugplot import DEBUGPLOT_CODES


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
        Debugging level for messages and plots. For details see
        'numina.array.display.pause_debugplot.py'.

    Returns
    -------
    sp_filtmask : numpy array
        Filtered and masked spectrum

    """

    # Fourier filtering
    xf = np.fft.fftfreq(sp.size)
    yf = np.fft.fft(sp)
    if abs(debugplot) in (21, 22):
        ximplotxy(xf, yf.real, xlim=(0, 0.51),
                  plottype='semilog', debugplot=debugplot)

    cut = (np.abs(xf) > fmax)
    yf[cut] = 0.0
    cut = (np.abs(xf) < fmin)
    yf[cut] = 0.0
    if abs(debugplot) in (21, 22):
        ximplotxy(xf, yf.real, xlim=(0, 0.51),
                  plottype='semilog', debugplot=debugplot)

    sp_filt = np.fft.ifft(yf).real
    if abs(debugplot) in (21, 22):
        ximplot(sp_filt, title="filtered median spectrum",
                plot_bbox=(1, sp_filt.size), debugplot=debugplot)

    sp_filtmask = sp_filt * cosinebell(sp_filt.size, 0.1)
    if abs(debugplot) in (21, 22):
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


def process_twilight(fitsfile, npix_zero_in_border,
                     oversampling, nwindow_median, outfile, debugplot):
    """Process twilight image.

    Parameters
    ----------
    fitsfile : str
        Twilight RSS FITS file name.
    npix_zero_in_border : int
        Number of pixels to be set to zero at the beginning and at
        the end of each spectrum to avoid unreliable pixel values
        produced in the wavelength calibration procedure.
    oversampling : int
        Oversampling of each pixel.
    nwindow_median : int
        Window size (in pixels) for median filter applied along the
        spectral direction.
    outfile : file
        Output FITS file name.
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
                title='initial twilight image', debugplot=debugplot)

    # set to zero a few pixels at the beginning and at the end of each
    # spectrum to avoid unreliable values coming from the wavelength
    # calibration procedure
    image2d = fix_pix_borders(image2d, nreplace=npix_zero_in_border,
                              sought_value=0, replacement_value=0)
    if abs(debugplot) in (21, 22):
        ximshow(image2d, show=True,
                title='twilight image after removing ' +
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
    # filtered and masked median spectrum
    spmedian_filtmask = filtmask(spmedian, fmin=0.02, fmax=0.15,
                                 debugplot=debugplot)
    if abs(debugplot) in (21, 22):
        xdum = np.arange(naxis1) + 1
        ax = ximplotxy(xdum, spmedian, show=False,
                       title="median spectrum", label='initial')
        ax.plot(xdum, spmedian_filtmask, label='filtered & masked')
        ax.legend()
        pause_debugplot(debugplot, pltshow=True)

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
        if i == naxis2_half and (abs(debugplot) in (21, 22)):
            ximplot(sp_filtmask, title="median spectrum of scan " + str(i),
                    plot_bbox=(1, spmedian.size), debugplot=debugplot)
        corr = periodic_corr1d(sp_filtmask, spmedian_filtmask)
        corr = corr[isort]
        if i == naxis2_half and (abs(debugplot) in (21, 22)):
            ximplotxy(xcorr, corr,
                      title="periodic correlation with scan " + str(i),
                      xlim=(-20, 20), debugplot=debugplot)
        ixpeak = np.array([corr.argmax()])
        xdum, sdum = refine_peaks_spectrum(corr, ixpeak, 7,
                                           method='gaussian')
        offsets[i] = xdum - naxis1_half

    if abs(debugplot) in (21, 22):
        xdum = np.arange(naxis2) + 1
        ax = ximplotxy(xdum, offsets, ylim=(-10, 10),
                       xlabel='pixel in the NAXIS2 direction',
                       ylabel='offset (pixels) in the NAXIS1 direction',
                       show=False, **{'label': 'measured offsets'})
        ypol1, residuals, reject = polfit_residuals_with_sigma_rejection(
            x=xdum, y=offsets, deg=1, times_sigma_reject=5.0)
        ax.plot(xdum, ypol1(xdum), '-',
                label='poly1: ' + str(ypol1.coef))
        ypol2, residuals, reject = polfit_residuals_with_sigma_rejection(
            x=xdum, y=offsets, deg=2, times_sigma_reject=5.0)
        ax.plot(xdum, ypol2(xdum), '-',
                label='poly2: ' + str(ypol2.coef))
        ax.legend()
        pause_debugplot(debugplot, pltshow=True)

    # oversampling
    naxis1_over = naxis1 * oversampling
    image2d_over = np.zeros((naxis2, naxis1_over))
    for i in range(naxis2):
        sp_over, crval1_over, cdelt1_over = \
            oversample1d(image2d[i, :], crval1=1, cdelt1=1,
                         oversampling=oversampling)
        sp_over_shifted = \
            shiftx_image2d_flux(sp_over, -offsets[i]*oversampling)
        image2d_over[i] = sp_over_shifted

    if abs(debugplot) in (21, 22):
        ximshow(image2d_over, title='oversampled & shifted',
                debugplot=debugplot)

    # medium spectrum (masking null values)
    image2d_over_masked = np.ma.masked_array(image2d_over,
                                             mask=(image2d_over == 0))
    spmedian_over = np.ma.median(image2d_over_masked, axis=0).data

    if abs(debugplot) in (21, 22):
        xplot = np.linspace(1, naxis1, naxis1)
        xplot_over = np.linspace(1, naxis1, naxis1_over)
        ax = ximplotxy(xplot, spmedian,
                       show=False, label='median')
        ax.plot(xplot_over, spmedian_over, 'b-', label='oversampled median')
        ax.legend()
        pause_debugplot(debugplot, pltshow=True)

    spmedian_over2d = \
        np.tile(spmedian_over, naxis2).reshape(naxis2, naxis1_over)
    image2d_over_norm = np.ones(image2d_over.shape)
    nonzero = np.where(spmedian_over2d != 0)
    image2d_over_norm[nonzero] = \
        image2d_over[nonzero] / spmedian_over2d[nonzero]
    image2d_divided = rebin(image2d_over_norm, naxis2, naxis1)

    # enlarge original mask to remove additional border effects
    mask2d = fix_pix_borders(mask2d, nreplace=npix_zero_in_border,
                             sought_value=True, replacement_value=True)
    image2d_divided[mask2d] = 1.0

    # apply median filter along the spectral direction to each spectrum
    # avoiding the masked region at the borders
    image2d_smoothed = np.ones((naxis2, naxis1))
    for i in range(naxis2):
        jmin, jmax = find_pix_borders(mask2d[i, :], sought_value=True)
        if jmin == -1 and jmax == naxis1:
            image2d_smoothed[i, :] = image2d_divided[i, :]
        else:
            j1 = max(jmin, 0)
            j2 = min(jmax, naxis1 - 1) + 1
            spdum = np.copy(image2d_divided[i, j1:j2])
            if j2 - j1 > nwindow_median:
                spfilt = ndimage.median_filter(spdum, nwindow_median,
                                               mode='nearest')
                image2d_smoothed[i, j1:j2] = spfilt
            else:
                image2d_smoothed[i, j1:j2] = spdum

    # residuals and robust standard deviation (using a masked array)
    image2d_residuals = np.ma.masked_array(image2d_divided - image2d_smoothed,
                                           mask=mask2d)
    q25, q75 = np.percentile(image2d_residuals.compressed(), q=[25.0, 75.0])
    sigma_g = 0.7413 * (q75 - q25)  # robust standard deviation

    # repeat median filter along the spectral direction to each spectrum
    # replacing suspicious pixels (residuals > 3*sigma_g) by a highly
    # smoothed version of the spectrum
    image2d_smoothed = np.ones((naxis2, naxis1))
    tsigma = 3.0
    ntmedian = 5
    for i in range(naxis2):
        jmin, jmax = find_pix_borders(mask2d[i, :], sought_value=True)
        if jmin == -1 and jmax == naxis1:
            image2d_smoothed[i, :] = image2d_divided[i, :]
        else:
            j1 = max(jmin, 0)
            j2 = min(jmax, naxis1 - 1) + 1
            spdum = np.copy(image2d_divided[i, j1:j2])
            if j2 - j1 > ntmedian*nwindow_median:
                spultrasmooth = ndimage.median_filter(spdum,
                                                      ntmedian*nwindow_median,
                                                      mode='nearest')
                spresiduals = image2d_residuals[i, j1:j2]
                pixreplace = np.where(np.abs(spresiduals) > tsigma*sigma_g)
                spdum[pixreplace] = spultrasmooth[pixreplace]
                spfilt = ndimage.median_filter(spdum, nwindow_median,
                                               mode='nearest')
                image2d_smoothed[i, j1:j2] = spfilt

            else:
                image2d_smoothed[i, j1:j2] = spdum

    if abs(debugplot) in (21, 22):
        ximshow(image2d_over_norm, title='divided (oversampled)',
                debugplot=debugplot)
        ximshow(image2d_divided,
                title='divided (resampled to original sampling)',
                debugplot=debugplot)
        ximshow(image2d_smoothed, title='median filtered (twice)',
                debugplot=debugplot)

    # save result
    hdu = fits.PrimaryHDU(image2d_smoothed, image2d_header)
    hdu.writeto(outfile, overwrite=True)


def main(args=None):
    # parse command-line options
    parser = argparse.ArgumentParser(prog='twilight')
    # positional parameters
    parser.add_argument("fitsfile",
                        help="Twilight FITS image",
                        type=argparse.FileType('r'))
    parser.add_argument("outfile",
                        help="Output FITS file name",
                        type=argparse.FileType('w'))
    parser.add_argument("--oversampling",
                        help="Oversampling (1=none; default)",
                        default=1, type=int)
    parser.add_argument("--npixzero",
                        help="Number of pixels to be set to zero at the "
                             "borders of each spectrum (default=3)",
                        default=3, type=int)
    parser.add_argument("--nwinmed",
                        help="Window size (pixels) for median filter along "
                             "the spectral direction (odd number, "
                             "default=51)",
                        default=51, type=int)
    parser.add_argument("--debugplot",
                        help="integer indicating plotting/debugging" +
                             " (default=0)",
                        type=int, default=0,
                        choices=DEBUGPLOT_CODES)

    args = parser.parse_args(args=args)

    process_twilight(args.fitsfile.name,
                     args.npixzero,
                     args.oversampling,
                     args.nwinmed,
                     args.outfile, args.debugplot)


if __name__ == "__main__":

    main()
