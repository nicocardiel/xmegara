from __future__ import division
from __future__ import print_function

import argparse
import astropy.io.fits as fits
import numpy as np
import os

from numina.array.display.pause_debugplot import pause_debugplot
from numina.array.display.polfit_residuals import polfit_residuals
from numina.array.display.ximplot import ximplot
from numina.array.display.ximshow import ximshow
from numina.array.wavecalib.peaks_spectrum import find_peaks_spectrum
from numina.array.wavecalib.peaks_spectrum import refine_peaks_spectrum

from numina.array.display.pause_debugplot import DEBUGPLOT_CODES

from fix_borders_wlcalib_rss import fix_pix_borders
from refine_master_wlcalib import match_wv_arrays


def fun_wv(xchannel, crpix1, crval1, cdelt1):
    """Compute wavelengths from channels.

    The wavelength calibration is provided through the usual parameters
    CRPIX1, CRVAL1 and CDELT1.

    Parameters
    ----------
    xchannel : numpy array
        Input channels where the wavelengths will be evaluated.
    crpix1: float
        CRPIX1 keyword.
    crval1: float
        CRVAL1 keyword.
    cdelt1: float
        CDELT1 keyword.

    Returns
    -------
    wv : numpy array
        Computed wavelengths

    """
    wv = crval1 + (xchannel - crpix1) * cdelt1
    return wv


def process_rss(fitsfile, linelist, npix_zero_in_border,
                geometry, debugplot):
    """Process twilight image.

    Parameters
    ----------
    fitsfile : str
        Wavelength calibrated RSS FITS file name.
    linelist : file handler
        ASCII file with the detailed list of expected arc lines.
    npix_zero_in_border : int
        Number of pixels to be set to zero at the beginning and at
        the end of each spectrum to avoid unreliable pixel values
        produced in the wavelength calibration procedure.
    geometry : tuple (4 integers) or None
        x, y, dx, dy values employed to set the Qt backend geometry.
    debugplot : int
        Debugging level for messages and plots. For details see
        'numina.array.display.pause_debugplot.py'.

    """

    # read the 2d image
    with fits.open(fitsfile) as hdulist:
        image2d_header = hdulist[0].header
        image2d = hdulist[0].data
    naxis2, naxis1 = image2d.shape
    crpix1 = image2d_header['crpix1']
    crval1 = image2d_header['crval1']
    cdelt1 = image2d_header['cdelt1']
    print('* Input file:', fitsfile)
    print('>>> NAXIS1:', naxis1)
    print('>>> NAXIS2:', naxis2)
    print('>>> CRPIX1:', crpix1)
    print('>>> CRVAL1:', crval1)
    print('>>> CDELT1:', cdelt1)

    if abs(debugplot) in (21, 22):
        ximshow(image2d, show=True,
                title='Wavelength calibrated RSS image', debugplot=debugplot)

    # set to zero a few pixels at the beginning and at the end of each
    # spectrum to avoid unreliable values coming from the wavelength
    # calibration procedure
    image2d = fix_pix_borders(image2d, nreplace=npix_zero_in_border,
                              sought_value=0, replacement_value=0)
    if abs(debugplot) in (21, 22):
        ximshow(image2d, show=True,
                title='RSS image after removing ' +
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

    # find initial line peaks
    nwinwidth_initial = 7
    ixpeaks = find_peaks_spectrum(spmedian, nwinwidth=nwinwidth_initial)

    # refine location of line peaks
    nwinwidth_refined = 5
    fxpeaks, sxpeaks = refine_peaks_spectrum(
        spmedian, ixpeaks,
        nwinwidth=nwinwidth_refined,
        method="gaussian"
    )

    ixpeaks_wv = fun_wv(ixpeaks + 1, crpix1, crval1, cdelt1)
    fxpeaks_wv = fun_wv(fxpeaks + 1, crpix1, crval1, cdelt1)

    # read list of expected arc lines
    master_table = np.genfromtxt(linelist)
    wv_master = master_table[:, 0]
    if abs(debugplot) in (21,22):
        print('wv_master:', wv_master)

    # match peaks with expected arc lines
    delta_wv_max = 2 * cdelt1
    wv_verified_all_peaks = match_wv_arrays(
        wv_master,
        fxpeaks_wv,
        delta_wv_max=delta_wv_max
    )
    lines_ok = np.where(wv_verified_all_peaks > 0)

    # compute residuals
    xresid = fxpeaks_wv[lines_ok]
    yresid = wv_verified_all_peaks[lines_ok] - fxpeaks_wv[lines_ok]

    # fit polynomial to residuals
    polyres, yresres = polfit_residuals(
        x=xresid,
        y=yresid,
        deg=1,
        use_r=True,
        debugplot=10
    )

    print('-' * 79)
    print(">>> Number of arc lines in master file:", len(wv_master))
    print(">>> Number of line peaks found........:", len(ixpeaks))
    print(">>> Number of identified lines........:", len(lines_ok[0]))
    list_wv_found = [str(round(wv, 4))
                     for wv in wv_verified_all_peaks if wv != 0]
    list_wv_master = [str(round(wv, 4)) for wv in wv_master]
    missing_wv = list(set(list_wv_master).symmetric_difference(set(list_wv_found)))
    print(">>> Unmatched lines...................:", missing_wv)

    # display results
    if abs(debugplot) % 10 != 0:
        from numina.array.display.matplotlib_qt import plt
        fig = plt.figure()
        if geometry is not None:
            x_geom, y_geom, dx_geom, dy_geom = geometry
            mngr = plt.get_current_fig_manager()
            mngr.window.setGeometry(x_geom, y_geom, dx_geom, dy_geom)

        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, 'Wavelength (Angstroms)',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=12)
        ax.axis('off')

        # residuals
        ax2 = fig.add_subplot(2, 1, 1)
        ax2.plot(xresid, yresid, 'o')
        ax2.set_ylabel('Residuals (Angstroms)')
        ax2.set_title('fitsfile: ' + os.path.basename(fitsfile) + '\n' +
                      'linelist: ' + os.path.basename(linelist.name),
                      **{'size': 10})
        xwv = fun_wv(np.arange(naxis1) + 1.0, crpix1, crval1, cdelt1)
        ax2.plot(xwv, polyres(xwv), '-')

        # median spectrum and peaks
        xmin = min(xwv)
        xmax = max(xwv)
        dx = xmax - xmin
        xmin -= dx / 80
        xmax += dx / 80
        ymin = min(spmedian)
        ymax = max(spmedian)
        dy = ymax - ymin
        ymin -= dy/20
        ymax += dy/20
        ax1 = fig.add_subplot(2, 1, 2, sharex=ax2)
        ax1.set_xlim([xmin, xmax])
        ax1.set_ylim([ymin, ymax])
        ax1.plot(xwv, spmedian)
        ax1.plot(ixpeaks_wv, spmedian[ixpeaks], 'o',
                label="initial location")
        ax1.plot(fxpeaks_wv, spmedian[ixpeaks], 'o',
                label="refined location")
        ax1.set_ylabel('Counts')
        ax1.xaxis.tick_top()
        ax1.xaxis.set_label_position('top')
        for i in range(len(ixpeaks)):
            if wv_verified_all_peaks[i] > 0:
                ax1.text(fxpeaks_wv[i], spmedian[ixpeaks[i]],
                        wv_verified_all_peaks[i], fontsize=8,
                        horizontalalignment='center')
            else:
                estimated_wv=fun_wv(fxpeaks[i] + 1, crpix1, crval1, cdelt1)
                estimated_wv=str(round(estimated_wv, 4))
                ax1.text(fxpeaks_wv[i], 0, # spmedian[ixpeaks[i]],
                         estimated_wv, fontsize=8, color='grey',
                         rotation='vertical',
                         horizontalalignment='center',
                         verticalalignment='top')
        if len(missing_wv) > 0:
            tmp = [float(wv) for wv in missing_wv]
            ax1.vlines(tmp, ymin=ymin, ymax=ymax,
                       colors='grey', linestyles='dotted',
                       label='missing lines')
        ax1.legend()
        pause_debugplot(debugplot, pltshow=True, tight_layout=True)


def main(args=None):
    # parse command-line options
    parser = argparse.ArgumentParser(prog='twilight')
    # positional parameters
    parser.add_argument("fitsfile",
                        help="Wavelength calibrated RSS FITS image",
                        type=argparse.FileType('r'))
    parser.add_argument("linelist",
                        help="ASCII file with detailed list of expected "
                             "arc lines",
                        type=argparse.FileType('r'))
    parser.add_argument("--npixzero",
                        help="Number of pixels to be set to zero at the "
                             "borders of each spectrum (default=3)",
                        default=3, type=int)
    parser.add_argument("--geometry",
                        help="tuple x,y,dx,dy",
                        default="0,0,640,480")
    parser.add_argument("--debugplot",
                        help="integer indicating plotting/debugging" +
                             " (default=0)",
                        type=int, default=12,
                        choices=DEBUGPLOT_CODES)

    args = parser.parse_args(args=args)

    # geometry
    if args.geometry is None:
        geometry = None
    else:
        tmp_str = args.geometry.split(",")
        x_geom = int(tmp_str[0])
        y_geom = int(tmp_str[1])
        dx_geom = int(tmp_str[2])
        dy_geom = int(tmp_str[3])
        geometry = x_geom, y_geom, dx_geom, dy_geom

    process_rss(args.fitsfile.name,
                args.linelist,
                args.npixzero,
                geometry=geometry,
                debugplot=args.debugplot)


if __name__ == "__main__":

    main()
