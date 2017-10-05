from __future__ import division
from __future__ import print_function

import json
import numpy as np

from numina.array.display.ximplotxy import ximplotxy
from numina.array.display.polfit_residuals import polfit_residuals
from numina.array.display.polfit_residuals import polfit_residuals_with_sigma_rejection
from numina.array.display.polfit_residuals import polfit_residuals_with_cook_rejection
from numina.array.display.pause_debugplot import pause_debugplot

master_wlcalib_file = 'master_wlcalib.json'
megadict = json.loads(open(master_wlcalib_file).read())
total_fibers = megadict['total_fibers']
contents_list = megadict['contents']

fibid = np.array([contents['fibid'] for contents in contents_list])
crval = np.array([contents['solution']['cr_linear']['crval'] for
                  contents in contents_list])
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
    poly, yres, reject= polfit_residuals_with_sigma_rejection(
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
        print('coeff a_'+ str(i) + ': ', sum(reject_all))
    else:
        # add new bad fits
        reject_all = np.logical_or(reject_all, reject)
        print('coeff a_'+ str(i) + ': ', sum(reject_all))

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
