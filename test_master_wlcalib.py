from __future__ import division
from __future__ import print_function

import argparse
from astropy.io import fits
import json
import numpy as np
from numpy.polynomial import Polynomial
import os
from uuid import uuid4

from numina.array.display.polfit_residuals import polfit_residuals
from numina.array.display.polfit_residuals import \
    polfit_residuals_with_sigma_rejection
from numina.array.display.pause_debugplot import pause_debugplot
from numina.array.display.ximplotxy import ximplotxy
from numina.array.wavecalib.arccalibration import fit_list_of_wvfeatures
from numina.array.wavecalib.peaks_spectrum import find_peaks_spectrum
from numina.array.wavecalib.peaks_spectrum import refine_peaks_spectrum
from numina.array.wavecalib.solutionarc import WavecalFeature

from numina.array.display.pause_debugplot import DEBUGPLOT_CODES


def filter_bad_fits(wlcalib_file, times_sigma_reject, debugplot):
    """Exctract useful information from master_wlcalib.

    Obtain the variation of each coefficient of the wavelength
    calibration polynomial as a function of the fiber number (assuming
    that the first fiber is fibid=1 and not 0).

    Parameters
    ----------
    wlcalib_file : file handler
        JSON file containing the initial wavelength calibration.
    times_sigma_reject : float
        Times sigma to reject points in fits.
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

    megadict = json.loads(open(wlcalib_file.name).read())
    contents_list = megadict['contents']

    fibid = np.array([contents['fibid'] for contents in contents_list])
    poldeg = [len(contents['solution']['coeff']) for contents in contents_list]
    if len(set(poldeg)) == 1:
        poldeg = poldeg[0] - 1
    else:
        raise ValueError("Non unique polynomial degree!")
    if abs(debugplot) >= 10:
        print('Polynomial degree:', poldeg)

    # determine bad fits from each independent polynomial coefficient
    for i in range(poldeg + 1):
        coeff = np.array([contents['solution']['coeff'][i] for
                          contents in contents_list])
        poly, yres, reject = polfit_residuals_with_sigma_rejection(
            x=fibid,
            y=coeff,
            deg=5,
            times_sigma_reject=times_sigma_reject,
        )
        if abs(debugplot) % 10 != 0:
            polfit_residuals(x=fibid, y=coeff, deg=5, reject=reject,
                             xlabel='fibid',
                             ylabel='coeff a_' + str(i),
                             title='Identifying bad fits',
                             debugplot=debugplot)
        if i == 0:
            reject_all = np.copy(reject)
            if abs(debugplot) >= 10:
                print('coeff a_' + str(i) + ': ', sum(reject_all))
        else:
            # add new bad fits
            reject_all = np.logical_or(reject_all, reject)
            if abs(debugplot) >= 10:
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


def match_wv_arrays(wv_master, wv_expected_all_peaks, delta_wv_max):
    """Verify expected wavelength for each line peak.

    Assign individual arc lines from wv_master to each expected
    wavelength when the latter is within the maximum allowed range.

    Parameters
    ----------
    wv_master : numpy array
        Array containing the master wavelengths.
    wv_expected_all_peaks : numpy array
        Array containing the expected wavelengths computed from the
        approximate polynomial calibration applied to the location of
        the line peaks.
    delta_wv_max : float
        Maximum distance to accept that the master wavelength
        corresponds to the expected wavelength.

    Returns
    -------
    wv_verified_all_peaks : numpy array
        Verified wavelengths from master list.

    """

    # initialize the output array to zero
    wv_verified_all_peaks = np.zeros_like(wv_expected_all_peaks)

    # initialize to True array to indicate that no peak has already
    # been verified (this flag avoids duplication)
    wv_unused = np.ones_like(wv_expected_all_peaks, dtype=bool)

    # since it is likely that len(wv_master) < len(wv_expected_all_peaks),
    # it is more convenient to execute the search in the following order
    for i in range(len(wv_master)):
        j = np.searchsorted(wv_expected_all_peaks, wv_master[i])
        if j == 0:
            if wv_unused[j]:
                delta_wv = abs(wv_master[i] - wv_expected_all_peaks[j])
                if delta_wv < delta_wv_max:
                    wv_verified_all_peaks[j] = wv_master[i]
                    wv_unused[j] = False
        elif j == len(wv_expected_all_peaks):
            if wv_unused[j-1]:
                delta_wv = abs(wv_master[i] - wv_expected_all_peaks[j-1])
                if delta_wv < delta_wv_max:
                    wv_verified_all_peaks[j-1] = wv_master[i]
                    wv_unused[j-1] = False
        else:
            delta_wv1 = abs(wv_master[i] - wv_expected_all_peaks[j-1])
            delta_wv2 = abs(wv_master[i] - wv_expected_all_peaks[j])
            if delta_wv1 < delta_wv2:
                if delta_wv1 < delta_wv_max:
                    if wv_unused[j-1]:
                        wv_verified_all_peaks[j-1] = wv_master[i]
                        wv_unused[j-1] = False
                    elif wv_unused[j]:
                        if delta_wv2 < delta_wv_max:
                            wv_verified_all_peaks[j] = wv_master[i]
                            wv_unused[j] = False
            else:
                if delta_wv2 < delta_wv_max:
                    if wv_unused[j]:
                        wv_verified_all_peaks[j] = wv_master[i]
                        wv_unused[j] = False
                    elif wv_unused[j-1]:
                        if delta_wv1 < delta_wv_max:
                            wv_verified_all_peaks[j-1] = wv_master[i]
                            wv_unused[j-1] = False

    return wv_verified_all_peaks


def refine_wlcalib(arc_rss, linelist, poldeg, list_poly, npix=2,
                   times_sigma_reject=5, debugplot=0):
    """Refine wavelength calibration using expected polynomial in each fiber.

    Parameters
    ----------
    arc_rss : file handler
        FITS file containing the uncalibrated RSS.
    linelist : file handler
        ASCII file with the detailed list of expected arc lines.
    poldeg : int
        Polynomial degree (must be the same for all the fibers).
    list_poly: list of polynomial instances
        List containing the polynomial variation of each wavelength
        calibration polynomial coefficient as a function of the fiber
        number.
    npix : int
        Number of pixels around each peak where the expected wavelength
        must match the tabulated wavelength in the master list.
    times_sigma_reject : float
        Times sigma to reject points in fits.
    debugplot : int
        Debugging level for messages and plots. For details see
        'numina.array.display.pause_debugplot.py'.

    Returns
    -------
    missing_fibers : list
        List of missing fibers
    contents : list of dictionaries
        Contents including the refined wavelength calibration. This
        list should replace the 'contents' section of the JSON file
        containing the master_wlcalib results.

    """

    # read input FITS file
    hdulist = fits.open(arc_rss)
    image2d = hdulist[0].data
    hdulist.close()

    naxis2, naxis1 = image2d.shape
    if abs(debugplot) >= 10:
        print('>>> Reading file:', arc_rss.name)
        print('>>> NAXIS1:', naxis1)
        print('>>> NAXIS2:', naxis2)

    # read list of expected arc lines
    master_table = np.genfromtxt(linelist)
    wv_master = master_table[:, 0]
    if abs(debugplot) >= 10:
        print('wv_master:', wv_master)

    # abscissae for plots
    xp = np.arange(1, naxis1 + 1)

    # initialized output lists
    missing_fibers = []
    contents = []

    # loop in naxis2
    for ifib in range(1, naxis2 + 1):
        sp = image2d[ifib - 1, :]

        # find initial line peaks
        nwinwidth_initial = 7
        ixpeaks = find_peaks_spectrum(sp, nwinwidth=nwinwidth_initial)

        # check there are enough lines for fit
        if len(ixpeaks) <= poldeg:
            print('WARNING: fibid, number of peaks:', ifib, len(ixpeaks))
            missing_fibers.append(ifib)
        else:
            # refine location of line peaks
            nwinwidth_refined = 5
            fxpeaks, sxpeaks = refine_peaks_spectrum(
                sp, ixpeaks,
                nwinwidth=nwinwidth_refined,
                method="gaussian"
            )
            if abs(debugplot) >= 10:
                print(">>> Number of lines found:", len(fxpeaks))

            # expected wavelength calibration polynomial for current fiber
            coeff = np.zeros(poldeg + 1)
            for k in range(poldeg + 1):
                dumpol = list_poly[k]
                coeff[k] = dumpol(ifib)
            wlpol = Polynomial(coeff)
            if abs(debugplot) >= 10:
                print(">>> Expected calibration polynomial:", wlpol)

            # expected wavelength of all identified peaks
            xchannel = fxpeaks + 1.0
            wv_expected_all_peaks = wlpol(xchannel)
            if abs(debugplot) in [21, 22]:
                for dum in zip(xchannel, wv_expected_all_peaks):
                    print('x, w:', dum)
                pause_debugplot(debugplot)

            # assign individual arc lines from master list to spectrum
            # line peaks when the expected wavelength is within the maximum
            # allowed range (+/- npix around the peak)
            crmin1_linear = wlpol(1)
            crmax1_linear = wlpol(naxis1)
            cdelt1_linear = (crmax1_linear - crmin1_linear) / (naxis1 - 1)
            delta_wv_max = npix * cdelt1_linear
            # iteration #1: find overall offset
            wv_verified_all_peaks = match_wv_arrays(
                wv_master,
                wv_expected_all_peaks,
                delta_wv_max=delta_wv_max
            )
            lines_ok = np.where(wv_verified_all_peaks > 0)
            wv_offsets_all_peaks = wv_verified_all_peaks-wv_expected_all_peaks
            overall_offset = np.median(wv_offsets_all_peaks[lines_ok])
            # iteration #2: use previous overall offset
            wv_expected_all_peaks += overall_offset
            wv_verified_all_peaks = match_wv_arrays(
                wv_master,
                wv_expected_all_peaks,
                delta_wv_max=delta_wv_max
            )

            # fit with sigma rejection
            lines_ok = np.where(wv_verified_all_peaks > 0)
            xdum = (fxpeaks + 1.0)[lines_ok]
            ydum = wv_verified_all_peaks[lines_ok]
            poly, yres, reject = polfit_residuals_with_sigma_rejection(
                x=xdum,
                y=ydum,
                deg=poldeg,
                times_sigma_reject=times_sigma_reject,
                debugplot=debugplot
            )
            # effective number of points
            npoints_eff = np.sum(np.logical_not(reject))
            # residual standard deviation
            sum_res2 = np.sum(yres[np.logical_not(reject)]**2)
            residual_std = np.sqrt(sum_res2/(npoints_eff - poldeg - 1))
            if True:  # abs(debugplot) >= 10:
                print("ifib, npoints_eff, residual_std:",
                      ifib, npoints_eff, residual_std)
                print("      poly.coef:", poly.coef)

            # generate dictionary with results associated with current fiber
            crmin1_linear = poly(1)
            crmax1_linear = poly(naxis1)
            cdelt1_linear = (crmax1_linear - crmin1_linear) / (naxis1 - 1)
            dumdict = {
                'fibid': ifib,
                'solution': {
                    'cr_linear': {
                        'crpix': 1.0,
                        'crmin': crmin1_linear,
                        'crmax': crmax1_linear,
                        'crval': crmin1_linear,
                        'cdelt': cdelt1_linear
                    },
                    'coeff': [poly.coef[k] for k in range(poldeg + 1)],
                    'features': [],
                    'npoints_eff': npoints_eff,
                    'residual_std': residual_std
                }
            }
            contents.append(dumdict)

            """
            # generate list of features
            xdum = xdum[np.logical_not(reject)]
            ydum = ydum[np.logical_not(reject)]
            list_of_wvfeatures = []
            for i in range(len(xdum)):
                wvfeature = WavecalFeature(
                    line_ok=True,
                    category="",
                    lineid=-1,
                    funcost=0.0,
                    xpos=xdum[i],
                    ypos=0.0,
                    peak=0.0,
                    fwhm=0.0,
                    reference=ydum[i]
                )
                list_of_wvfeatures.append(wvfeature)

            solution_wv = fit_list_of_wvfeatures(
                list_of_wvfeatures=list_of_wvfeatures,
                naxis1_arc=naxis1,
                crpix1=1.0,
                poly_degree_wfit=poldeg,
                weighted=False,
                debugplot=debugplot,
                plot_title=arc_rss.name + ' [fiber #' + str(ifib) + ']\n' +
                           linelist.name
            )
            if True:  #abs(debugplot) >= 10:
                print("ifib, solution_wv.coeff:\n", ifib, solution_wv.coeff)
            """

            # display spectrum and peaks
            if abs(debugplot) % 10 != 0:
                title = arc_rss.name + ' [fiber #' + str(ifib) + ']'
                ax = ximplotxy(xp, sp,
                               xlabel='pixel (from 1 to NAXIS1)',
                               ylabel='number of counts',
                               title=title,
                               show=False, debugplot=debugplot)
                ymin = sp.min()
                ymax = sp.max()
                dy = ymax - ymin
                ymin -= dy / 20.
                ymax += dy / 20.
                ax.set_ylim([ymin, ymax])
                # mark peak location
                ax.plot(ixpeaks + 1, sp[ixpeaks], 'bo',
                        label="initial location")
                ax.plot(fxpeaks + 1, sp[ixpeaks], 'go',
                        label="refined location")
                ax.plot((fxpeaks + 1)[lines_ok], sp[ixpeaks][lines_ok], 'mo',
                        label="identified lines")
                for i in range(len(ixpeaks)):
                    if wv_verified_all_peaks[i] > 0:
                        ax.text(fxpeaks[i] + 1.0, sp[ixpeaks[i]],
                                wv_verified_all_peaks[i], fontsize=8,
                                horizontalalignment='center')
                # legend
                ax.legend(numpoints=1)
                # show plot
                pause_debugplot(debugplot, pltshow=True)

    return missing_fibers, contents


def main(args=None):
    # parse command-line options
    parser = argparse.ArgumentParser(prog='overplot_traces')
    # positional parameters
    parser.add_argument("uncalibrated_arc_rss",
                        help="FITS image containing wavelength uncalibrated "
                             "RSS",
                        type=argparse.FileType('r'))
    parser.add_argument("wlcalib_file",
                        help="JSON file with initial wavelength calibration",
                        type=argparse.FileType('r'))
    parser.add_argument("linelist",
                        help="ASCII file with detailed list of expected "
                             "arc lines",
                        type=argparse.FileType('r'))
    parser.add_argument("--debugplot",
                        help="integer indicating plotting/debugging" +
                             " (default=0)",
                        type=int, default=0,
                        choices=DEBUGPLOT_CODES)
    args = parser.parse_args(args=args)

    poldeg, list_poly = filter_bad_fits(
        args.wlcalib_file,
        times_sigma_reject=5.0,
        debugplot=args.debugplot
    )

    missing_fibers, contents = refine_wlcalib(
        args.uncalibrated_arc_rss,
        args.linelist,
        poldeg,
        list_poly,
        npix=2,
        times_sigma_reject=5.0,
        debugplot=args.debugplot
    )

    megadict = json.loads(open(args.wlcalib_file.name).read())
    megadict['missing_fibers'] = missing_fibers
    megadict['contents'] = contents
    megadict['uuid'] = str(uuid4())
    outfile = os.path.basename(args.wlcalib_file.name) + '_refined'
    print("Generating: " + outfile)
    with open(outfile, 'w') as fstream:
        json.dump(megadict, fstream, indent=2, sort_keys=True)


if __name__ == "__main__":

    main()
