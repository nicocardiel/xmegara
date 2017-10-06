from __future__ import division
from __future__ import print_function

import argparse
from astropy.io import fits
import json
import numpy as np

from numina.array.display.polfit_residuals import polfit_residuals
from numina.array.display.polfit_residuals import \
    polfit_residuals_with_sigma_rejection
from numina.array.display.pause_debugplot import pause_debugplot
from numina.array.display.ximplotxy import ximplotxy
from numina.array.wavecalib.peaks_spectrum import find_peaks_spectrum
from numina.array.wavecalib.peaks_spectrum import refine_peaks_spectrum

from numina.array.display.pause_debugplot import DEBUGPLOT_CODES


def filter_bad_fits(master_wlcalib_file, debugplot):
    """Exctract useful information from master_wlcalib.

    Obtain the variation of each coefficient of the wavelength
    calibration polynomial as a function of the fiber number (assuming
    that the first fiber is fibid=1 and not 0).

    Parameters
    ----------
    master_wlcalib_file : file handler
        JSON file containing the initial wavelength calibration.
    debugplot : int
        Debugging level for messages and plots. For details see
        'numina.array.display.pause_debugplot.py'.

    Returns
    -------
    poldeg : int
        Polynomial degree (must be the same for all the fibers).
    list_poly: list of polynomial instances
        List containing the polynomial variation of each wavelength
        calibration polynomical coefficient as a function of the fiber
        number.

    """

    reject_all = [None]  # avoid PyCharm warning

    megadict = json.loads(open(master_wlcalib_file.name).read())
    contents_list = megadict['contents']

    fibid = np.array([contents['fibid'] for contents in contents_list])
    poldeg = [len(contents['solution']['coeff']) for contents in contents_list]
    if len(set(poldeg)) == 1:
        poldeg = poldeg[0] - 1
    else:
        raise ValueError("Non unique polynomial degree!")
    if debugplot >= 10:
        print('Polynomial degree:', poldeg)

    # determine bad fits from each independent polynomial coefficient
    for i in range(poldeg + 1):
        coeff = np.array([contents['solution']['coeff'][i] for
                          contents in contents_list])
        poly, yres, reject = polfit_residuals_with_sigma_rejection(
            x=fibid,
            y=coeff,
            deg=5,
            times_sigma_reject=5,
        )
        if debugplot % 10 != 0:
            polfit_residuals(x=fibid, y=coeff, deg=5, reject=reject,
                             xlabel='fibid',
                             ylabel='coeff a_' + str(i),
                             title='Identifying bad fits',
                             debugplot=debugplot)
        if i == 0:
            reject_all = np.copy(reject)
            if debugplot >= 10:
                print('coeff a_' + str(i) + ': ', sum(reject_all))
        else:
            # add new bad fits
            reject_all = np.logical_or(reject_all, reject)
            if debugplot >= 10:
                print('coeff a_' + str(i) + ': ', sum(reject_all))

    # determine new fits excluding all fibers with bad fits
    poly_list = []
    for i in range(poldeg + 1):
        coeff = np.array([contents['solution']['coeff'][i] for
                          contents in contents_list])
        poly, yres = polfit_residuals(
            x=fibid,
            y=coeff,
            deg=5,
            reject=reject_all,
            xlabel='fibid',
            ylabel='coeff a_' + str(i),
            title='Computing filtered fits',
            debugplot=debugplot
        )
        poly_list.append(poly)

    return poldeg, poly_list


def refine_wlcalib(arc_rss, poldeg, list_poly, debugplot=0):
    """Refine wavelength calibration using expected polynomial in each fiber.

    Parameters
    ----------
    arc_rss : file handler
        FITS file containing the uncalibrated RSS.
    poldeg : int
        Polynomial degree (must be the same for all the fibers).
    list_poly: list of polynomial instances
        List containing the polynomial variation of each wavelength
        calibration polynomical coefficient as a function of the fiber
        number.
    debugplot : int
        Debugging level for messages and plots. For details see
        'numina.array.display.pause_debugplot.py'.

    """

    # read input FITS file
    hdulist = fits.open(arc_rss)
    image2d = hdulist[0].data
    hdulist.close()

    naxis2, naxis1 = image2d.shape
    if debugplot >= 10:
        print('>>> Reading file:', arc_rss.name)
        print('>>> NAXIS1:', naxis1)
        print('>>> NAXIS2:', naxis2)

    xp = np.arange(1, naxis1 + 1)
    # loop in naxis2
    for ifib in range(1, naxis2 + 1):
        sp = image2d[ifib - 1,:]
        nwinwidth_initial = 7
        ixpeaks = find_peaks_spectrum(sp, nwinwidth=nwinwidth_initial)
        # check there are enough lines for fit
        if len(ixpeaks) <= poldeg:
            print('WARNING: fibid, number of peaks:', ifib, len(ixpeaks))
        else:
            nwinwidth_refined = 5
            fxpeaks, sxpeaks = refine_peaks_spectrum(
                sp, ixpeaks,
                nwinwidth=nwinwidth_refined,
                method="gaussian"
            )
            if debugplot >= 10:
                print(">>> Number of lines found:", len(fxpeaks))
            # display median spectrum and peaks
            if debugplot % 10 != 0:
                title = arc_rss.name + ' [fiber #' + str(ifib) + ']'
                ax = ximplotxy(xp, sp, title=title,
                               show=False, debugplot=debugplot)
                ymin = sp.min()
                ymax = sp.max()
                dy = ymax - ymin
                ymin -= dy / 20.
                ymax += dy / 20.
                ax.set_ylim([ymin, ymax])
                # mark peak location
                ax.plot(ixpeaks + 1,
                        sp[ixpeaks], 'bo', label="initial location")
                ax.plot(fxpeaks + 1,
                        sp[ixpeaks], 'go', label="refined location")
                # legend
                ax.legend(numpoints=1)
                # show plot
                pause_debugplot(debugplot, pltshow=True)


def main(args=None):
    # parse command-line options
    parser = argparse.ArgumentParser(prog='overplot_traces')
    # positional parameters
    parser.add_argument("uncalibrated_arc_rss",
                        help="FITS image containing uncalibrated RSS",
                        type=argparse.FileType('r'))
    parser.add_argument("wlcalib_file",
                        help="JSON file with initial wavelength calibration",
                        type=argparse.FileType('r'))
    parser.add_argument("linelist",
                        help="ASCII file with detailed list of identified "
                             "arc lines",
                        type=argparse.FileType('r'))
    parser.add_argument("--debugplot",
                        help="integer indicating plotting/debugging" +
                             " (default=0)",
                        type=int, default=0,
                        choices=DEBUGPLOT_CODES)
    args = parser.parse_args(args=args)

    poldeg, list_poly = filter_bad_fits(args.wlcalib_file, args.debugplot)
    refine_wlcalib(args.uncalibrated_arc_rss,
                   poldeg, list_poly, args.debugplot)


if __name__ == "__main__":

    main()
