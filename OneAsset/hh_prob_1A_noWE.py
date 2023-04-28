'''Standard Incomplete Market model'''

import numpy as np

from sequence_jacobian import het
from sequence_jacobian import interpolate, misc, grids

'''Core HetBlock'''

def hh_init(a_grid, we, r, eis):
    coh = (1 + r) * a_grid[np.newaxis, :] + we[:, np.newaxis]
    Va = (1 + r) * (0.1 * coh) ** (-1 / eis)
    return Va


@het(exogenous='Pi', policy='a', backward='Va', backward_init=hh_init)
def hh(Va_p, a_grid, we, r, beta, eis,cbar,T):
    uc_nextgrid = beta * Va_p
    c_nextgrid = uc_nextgrid ** (-eis)+cbar
    coh = (1 + r) * a_grid[np.newaxis, :] + we[:, np.newaxis]+ T[:, np.newaxis]
    a = interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    misc.setmin(a, a_grid[0])
    c = coh - a
    Va = (1 + r) * (c-cbar) ** (-1 / eis)*(c>cbar)+10e6*(c<=cbar)
    return Va, a, c
    