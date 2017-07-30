from __future__ import division
from __future__ import print_function

import argparse
import json
import numpy as np
from numpy.polynomial import Polynomial
import pkgutil
from uuid import uuid4

from numina.array.display.polfit_residuals import polfit_residuals
from numina.array.display.ximshow import ximshow_file
from numina.array.display.pause_debugplot import pause_debugplot


def assign_boxes_to_fibers(insmode):
    """Read boxes in configuration file and assign values to fibid

    Parameters
    ----------
    insmode : string
        Value of the INSMODE keyword: 'LCB' or 'MOS'.

    Returns
    -------
    fibid_with_box : list of strings
        List with string label that contains both the fibid and the
        box name.

    """
    sdum = pkgutil.get_data(
        'megaradrp.instrument.configs',
        'component-86a6e968-8d3d-456f-89f8-09ff0c7f7c57.json'
    )
    dictdum = json.loads(sdum)
    pseudo_slit_config = \
        dictdum['configurations']['boxes']['values'][insmode]

    fibid_with_box = []
    n1 = 1
    for dumbox in pseudo_slit_config:
        nfibers = dumbox['nfibers']
        name = dumbox['name']
        n2 = n1 + nfibers
        fibid_with_box += \
            ["{}  [{}]".format(val1, val2)
             for val1, val2 in zip(range(n1, n2), [name] * nfibers)]
        n1 = n2

    return fibid_with_box


def plot_trace(ax, coeff, xmin, xmax, ix_offset,
               rawimage, fibids, fiblabel, colour):
    if xmin == xmax == 0:
        num = 4096
        xp = np.linspace(start=1, stop=4096, num=num)
    else:
        num = int(float(xmax - xmin + 1) + 0.5)
        xp = np.linspace(start=xmin, stop=xmax, num=num)
    ypol = Polynomial(coeff)
    yp = ypol(xp)
    if rawimage:
        lcut = (yp > 2056.5)
        yp[lcut] += 100
    ax.plot(xp + ix_offset, yp + 1, color=colour, linestyle='dotted')
    if fibids:
        if xmin == xmax == 0:
            xmidpoint = 2048
        else:
            xmidpoint = (xmin+xmax)/2
        ax.text(xmidpoint, yp[int(num / 2)], fiblabel, fontsize=6,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="grey", ),
                color=colour, fontweight='bold', backgroundcolor='white',
                ha='center')


def main(args=None):
    # parse command-line options
    parser = argparse.ArgumentParser(prog='overplot_traces')
    # positional parameters
    parser.add_argument("fits_file",
                        help="FITS image containing the spectra",
                        type=argparse.FileType('r'))
    parser.add_argument("traces_file",
                        help="JSON file with fiber traces",
                        type=argparse.FileType('r'))
    # optional parameters
    parser.add_argument("--rawimage",
                        help="FITS file is a RAW image (RSS assumed instead)",
                        action="store_true")
    parser.add_argument("--yoffset",
                        help="Vertical offset (+upwards, -downwards)",
                        default=0,
                        type=float)
    parser.add_argument("--extrapolate",
                        help="Extrapolate traces plot",
                        action="store_true")
    parser.add_argument("--fibids",
                        help="Display fiber identification number",
                        action="store_true")
    parser.add_argument("--healing",
                        help="JSON healing file to improve traces",
                        type=argparse.FileType('r'))
    parser.add_argument("--updated_traces",
                        help="JSON file with modified fiber traces",
                        type=argparse.FileType('w'))
    parser.add_argument("--z1z2",
                        help="tuple z1,z2, minmax or None (use zscale)")
    parser.add_argument("--bbox",
                        help="bounding box tuple: nc1,nc2,ns1,ns2")
    parser.add_argument("--keystitle",
                        help="tuple of FITS keywords.format: " +
                             "key1,key2,...keyn.'format'")
    parser.add_argument("--geometry",
                        help="tuple x,y,dx,dy",
                        default="0,0,640,480")
    parser.add_argument("--pdffile",
                        help="ouput PDF file name",
                        type=argparse.FileType('w'))

    args = parser.parse_args(args=args)

    # read pdffile
    if args.pdffile is not None:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf = PdfPages(args.pdffile.name)
    else:
        import matplotlib
        matplotlib.use('Qt5Agg')
        pdf = None

    ax = ximshow_file(args.fits_file.name,
                      args_cbar_orientation='vertical',
                      args_z1z2=args.z1z2,
                      args_bbox=args.bbox,
                      args_keystitle=args.keystitle,
                      args_geometry=args.geometry,
                      pdf=pdf,
                      show=False)

    # trace offsets for RAW images
    if args.rawimage:
        ix_offset = 51
    else:
        ix_offset = 1

    # read and display traces from JSON file
    bigdict = json.loads(open(args.traces_file.name).read())
    insmode = bigdict['tags']['insmode']
    fibid_with_box = assign_boxes_to_fibers(insmode)
    total_fibers = bigdict['total_fibers']
    if total_fibers != len(fibid_with_box):
        raise ValueError('Mismatch between number of fibers and '
                         'expected number from account from boxes')
    global_offset = bigdict['global_offset']
    pol_global_offset = np.polynomial.Polynomial(global_offset)
    ref_column = bigdict['ref_column']
    for fiberdict in bigdict['contents']:
        fibid = fiberdict['fibid']
        fiblabel = fibid_with_box[fibid - 1]
        xmin = fiberdict['start']
        xmax = fiberdict['stop']
        coeff = np.array(fiberdict['fitparms'])
        # skip fibers without trace
        if len(coeff) > 0:
            pol_trace = np.polynomial.Polynomial(coeff)
            y_at_ref_column = pol_trace(ref_column)
            correction = pol_global_offset(y_at_ref_column)
            coeff[0] += correction + args.yoffset
            # update values in bigdict (JSON structure)
            bigdict['contents'][fibid-1]['fitparms'] = coeff.tolist()
            if args.extrapolate:
                plot_trace(ax, coeff, 0, 0, ix_offset, args.rawimage,
                           False, fiblabel, colour='grey')
            plot_trace(ax, coeff, xmin, xmax, ix_offset, args.rawimage,
                       args.fibids, fiblabel, colour='blue')
        else:
            print('Warning ---> Missing fiber:',
                  fibid, '-->', fibid_with_box[fibid - 1])

    # if present, read healing JSON file
    if args.healing is not None:
        healdict = json.loads(open(args.healing.name).read())
        list_operations = healdict['operations']
        for operation in list_operations:

            if operation['description'] == 'extrapolation':
                fibid_ini = operation['fibid_ini']
                fibid_end = operation['fibid_end']
                if 1 <= fibid_ini <= fibid_end <= total_fibers:
                    pass
                else:
                    raise ValueError('Invalid fibid_ini or fibid_end')
                for fibid in range(fibid_ini, fibid_end + 1):
                    fiblabel = fibid_with_box[fibid - 1]
                    if len(bigdict['contents'][fibid - 1]['fitparms']) > 0:
                        print('(extrapolation) fibid:', fibid, '-->', fiblabel)
                        # update values in bigdict (JSON structure)
                        start = operation['start']
                        stop = operation['stop']
                        start_orig = bigdict['contents'][fibid - 1]['start']
                        stop_orig = bigdict['contents'][fibid - 1]['stop']
                        bigdict['contents'][fibid - 1]['start'] = start
                        bigdict['contents'][fibid - 1]['stop'] = stop
                        coeff = np.array(
                            bigdict['contents'][fibid - 1]['fitparms']
                        )
                        if start < start_orig:
                            plot_trace(ax, coeff, start, start_orig,
                                       ix_offset,
                                       args.rawimage, False, fiblabel,
                                       colour='green')
                        if stop_orig < stop:
                            plot_trace(ax, coeff, stop_orig, stop,
                                       ix_offset,
                                       args.rawimage, False, fiblabel,
                                       colour='green')
                    else:
                        print('(extrapolation SKIPPED) fibid:',
                              fibid, '-->', fiblabel)

            elif operation['description'] == \
                    'extrapolation_through_fixed_points':
                fibid = operation['fibid']
                fiblabel = fibid_with_box[fibid - 1]
                print('(extrapolation+fixed):', fibid, '-->', fiblabel)
                start_reuse = operation['start_reuse']
                stop_reuse = operation['stop_reuse']
                resampling = operation['resampling']
                poldeg = operation['poldeg']
                start = operation['start']
                stop = operation['stop']
                coeff = bigdict['contents'][fibid - 1]['fitparms']
                xfit = np.linspace(start_reuse, stop_reuse, num=resampling)
                poly = np.polynomial.Polynomial(coeff)
                yfit = poly(xfit)
                for fixedpoint in operation['fixed_points']:
                    # assume x, y coordinates in JSON file are given in
                    # image coordinates, starting at (1,1) in the lower
                    # left corner
                    xdum = fixedpoint['x'] - 1  # use np.array coordinates
                    ydum = fixedpoint['y'] - 1  # use np.array coordinates
                    xfit = np.concatenate((xfit, np.array([xdum])))
                    yfit = np.concatenate((yfit, np.array([ydum])))
                poly, residum = polfit_residuals(xfit, yfit, poldeg)
                coeff = poly.coef
                if start < start_reuse:
                    plot_trace(ax, coeff, start, start_reuse, ix_offset,
                               args.rawimage, args.fibids, fiblabel,
                               colour='green')
                if stop_reuse < stop:
                    plot_trace(ax, coeff, stop_reuse, stop, ix_offset,
                               args.rawimage, args.fibids, fiblabel,
                               colour='green')
                bigdict['contents'][fibid - 1]['start'] = start
                bigdict['contents'][fibid - 1]['stop'] = stop
                bigdict['contents'][fibid - 1]['fitparms'] = coeff.tolist()

            elif operation['description'] == 'interpolation2':
                fibid = operation['fibid']
                fiblabel = fibid_with_box[fibid - 1]
                print('(interpolation2) fibid:', fibid, '-->', fiblabel)
                fraction = operation['fraction']
                nf1, nf2 = operation['neighbours']
                tmpf1 = bigdict['contents'][nf1 - 1]
                tmpf2 = bigdict['contents'][nf2 - 1]
                if nf1 != tmpf1['fibid'] or nf2 != tmpf2['fibid']:
                    raise ValueError(
                        "Unexpected fiber numbers in neighbours"
                    )
                coefff1 = np.array(tmpf1['fitparms'])
                coefff2 = np.array(tmpf2['fitparms'])
                xmin = np.min([tmpf1['start'], tmpf2['start']])
                xmax = np.min([tmpf1['stop'], tmpf2['stop']])
                coeff = coefff1 + fraction * (coefff2 - coefff1)
                plot_trace(ax, coeff, xmin, xmax, ix_offset,
                           args.rawimage, args.fibids,
                           fiblabel,
                           colour='green')
                # update values in bigdict (JSON structure)
                bigdict['contents'][fibid - 1]['start'] = xmin
                bigdict['contents'][fibid - 1]['stop'] = xmax
                bigdict['contents'][fibid - 1][
                    'fitparms'] = coeff.tolist()
                if fibid in bigdict['error_fitting']:
                    bigdict['error_fitting'].remove(fibid)

            else:
                raise ValueError('Unexpected healing method:',
                                 operation['description'])

# update trace map
    if args.updated_traces is not None:
        # avoid overwritting initial JSON file
        if args.updated_traces.name != args.traces_file.name:
            # new random uuid for the updated calibration
            bigdict['uuid'] = str(uuid4())
            with open(args.updated_traces.name, 'w') as outfile:
                json.dump(bigdict, outfile, indent=2)

    if pdf is not None:
        pdf.savefig()
        pdf.close()
    else:
        pause_debugplot(12, pltshow=True, tight_layout=True)


if __name__ == "__main__":

    main()
