from __future__ import division
from __future__ import print_function

import argparse
import json
import numpy as np

from numina.array.display.polfit_residuals import polfit_residuals
from numina.array.display.polfit_residuals import \
    polfit_residuals_with_sigma_rejection


def filter_bad_fits(master_wlcalib_file):
    """Exctract useful information from master_wlcalib.

    Obtain the variation of each coefficient of the wavelength
    calibration polynomial as a function of the fiber number (assuming
    that the first fiber is fibid=1 and not 0).

    Parameters
    ----------
    master_wlcalib_file : file handler
        JSON file containing the initial wavelength calibration.

    Returns
    -------
    poldeg : int
        Polynomial degree (must be the same for all the fibers).
    list_poly: list of polynomial instances
        List containing the polynomial variation of each wavelength
        calibration polynomical coefficient as a function of the fiber
        number

    """

    reject_all = [None]  # avoid PyCharm warning

    megadict = json.loads(open(master_wlcalib_file).read())
    contents_list = megadict['contents']

    fibid = np.array([contents['fibid'] for contents in contents_list])
    poldeg = [len(contents['solution']['coeff']) for contents in contents_list]
    if len(set(poldeg)) == 1:
        poldeg = poldeg[0] - 1
    else:
        raise ValueError("Non unique polynomial degree!")
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
        polfit_residuals(x=fibid, y=coeff, deg=5, reject=reject,
                         xlabel='fibid',
                         ylabel='coeff a_' + str(i),
                         title='Identifying bad fits',
                         debugplot=12)
        if i == 0:
            reject_all = np.copy(reject)
            print('coeff a_' + str(i) + ': ', sum(reject_all))
        else:
            # add new bad fits
            reject_all = np.logical_or(reject_all, reject)
            print('coeff a_' + str(i) + ': ', sum(reject_all))

    # determine new fits excluding fibers with bad fits
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
            debugplot=12
        )
        poly_list.append(poly)

    print(poly_list)

    return poldeg, poly_list


def main(args=None):
    # parse command-line options
    parser = argparse.ArgumentParser(prog='overplot_traces')
    # positional parameters
    parser.add_argument("uncalibrated_arc_rss",
                        help="FITS image containing uncalibrated RSS ",
                        type=argparse.FileType('r'))
    parser.add_argument("wlcalib_file",
                        help="JSON file with initial wavelength calibration",
                        type=argparse.FileType('r'))
    args = parser.parse_args(args=args)

    poldeg, list_poly = filter_bad_fits(args.wlcalib_file)


if __name__ == "__main__":

    main()
