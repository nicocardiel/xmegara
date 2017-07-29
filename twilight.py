from __future__ import division
from __future__ import print_function

import argparse
import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyval
from numina.array.display.pause_debugplot import pause_debugplot
from numina.array.display.polfit_residuals import \
    polfit_residuals_with_sigma_rejection
from numina.array.display.ximplot import ximplot
from numina.array.display.ximshow import ximshow
from numina.array.display.ximplotxy import ximplotxy
from numina.array.interpolation import SteffenInterpolator
from numina.array.wavecalib.peaks_spectrum import refine_peaks_spectrum
from scipy import ndimage

from numina.array.display.pause_debugplot import DEBUGPLOT_CODES


def find_pix_borders(sp, sought_value):
    """Find useful region of MEGARA spectrum.

    Detemine the useful region of a given MEGARA spectrum by skipping
    the initial (final) pixels with values equal to 'sought_value'.

    Parameters
    ----------
    sp : 1D numpy array
        MEGARA spectrum.
    sought_value : int, float, bool
        Pixel value that indicate missing data in the spectrum.

    Returns
    -------
    jmin, jmax : tuple (integers)
        Valid spectrum region (in array coordinates, from 0 to
        NAXIS1 - 1). If all the spectra is equal to 'sought_value',
        the returned values are jmin=-1 and jmax=naxis1.

    """

    if sp.ndim != 1:
        raise ValueError('Unexpected number of dimensions:', sp.ndim)
    naxis1 = len(sp)

    jborder_min = -1
    jborder_max = naxis1

    # only spectra with values different from 'sought_value'
    if not np.alltrue(sp == sought_value):
        # left border
        while True:
            jborder_min += 1
            if sp[jborder_min] != sought_value:
                break
        # right border
        while True:
            jborder_max -= 1
            if sp[jborder_max] != sought_value:
                break

    return jborder_min, jborder_max


def fix_pix_borders(image2d, nreplace, sought_value, replacement_value):
    """Replace a few pixels at the borders of each spectrum.

    Set to 'replacement_value' 'nreplace' pixels at the beginning (at
    the end) of each spectrum just after (before) the spectrum value
    changes from (to) 'sought_value', as seen from the image borders.

    Parameters
    ----------
    image2d : numpy array
        Initial 2D image.
    nreplace : int
        Number of pixels to be replaced in each border.
    sought_value : int, float, bool
        Pixel value that indicate missing data in the spectrum.
    replacement_value : int, float, bool
        Pixel value to be employed in the 'nreplace' pixels.

    Returns
    -------
    image2d : numpy array
        Final 2D image.

    """

    naxis2, naxis1 = image2d.shape

    for i in range(naxis2):
        # only spectra with values different from 'sought_value'
        jborder_min, jborder_max = find_pix_borders(image2d[i, :],
                                                    sought_value=sought_value)
        # left border
        if jborder_min != -1:
            j1 = jborder_min
            j2 = min(j1 + nreplace, naxis1)
            image2d[i, j1:j2] = replacement_value
        # right border
        if jborder_max != naxis1:
            j2 = jborder_max + 1
            j1 = max(j2 - nreplace, 0)
            image2d[i, j1:j2] = replacement_value

    return image2d


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


def oversample1d(sp, crval1, cdelt1, oversampling=1, debugplot=0):
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
    debugplot : int
        Debugging level for messages and plots. For details see
        'numina.array.display.pause_debugplot.py'.

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

    if abs(debugplot) in (21, 22):
        crvaln = crval1 + (naxis1 - 1) * cdelt1
        crvaln_over = crval1_over + (naxis1_over - 1) * cdelt1_over
        xover = np.linspace(crval1_over, crvaln_over, naxis1_over)
        ax = ximplotxy(np.linspace(crval1, crvaln, naxis1), sp, 'bo',
                       label='original', show=False)
        ax.plot(xover, sp_over, 'r+', label='resampled')
        pause_debugplot(debugplot, pltshow=True)

    return sp_over, crval1_over, cdelt1_over


def rebin(a, *args):
    """See http://scipy-cookbook.readthedocs.io/items/Rebinning.html

    Note: integer division in the computation of 'factor' has been
    included to avoid the following runtime message:
    VisibleDeprecationWarning: using a non-integer number instead of
    an integer will result in an error in the future
    from __future__ import division

    """
    shape = a.shape
    len_shape = len(shape)
    factor = np.asarray(shape) // np.asarray(args)
    ev_list = ['a.reshape('] + \
              ['args[%d], factor[%d], ' % (i, i) for i in range(len_shape)] + \
              [')'] + ['.mean(%d)' % (i+1) for i in range(len_shape)]
    # print(''.join(ev_list))
    return eval(''.join(ev_list))


def shiftx_image2d_flux(image2d_orig, xoffset):
    """Resample 2D image using a shift in the x direction (flux is preserved).

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
    elif image2d_orig.ndim == 2:
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
    """Resample a 1D/2D image using NAXIS1, CDELT1, CRVAL1, and CRPIX1.

    The same NAXIS1, CDELT1, CRVAL1, and CRPIX1 are employed for all
    the scans (rows) of the original 'image2d'. The wavelength
    calibrated output image has dimensions NAXIS1 * NSCAN, where NSCAN
    is the original number of scans (rows) of the original image.

    Flux is preserved.

    Parameters
    ----------
    image2d_orig : numpy array
        1D or 2D image to be resampled.
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
        Wavelength calibrated 1D or 2D image.

    """

    # duplicate input array, avoiding problems when using as input
    # 1d numpy arrays with shape (nchan,) instead of a 2d numpy
    # array with shape (1,nchan)
    if image2d_orig.ndim == 1:
        nscan = 1
        nchan = image2d_orig.size
        image2d = np.zeros((nscan, nchan))
        image2d[0, :] = np.copy(image2d_orig)
    elif image2d_orig.ndim == 2:
        nscan, nchan = image2d_orig.shape
        image2d = np.copy(image2d_orig)
    else:
        print('>>> image2d_orig.shape:', image2d_orig.shape)
        raise ValueError('Unexpected number of dimensions')

    new_x = np.arange(naxis1)
    new_wl = crval1 + cdelt1 * new_x

    old_x_borders = np.arange(-0.5, nchan)
    old_x_borders += crpix1  # following FITS criterium

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
