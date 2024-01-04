import numpy as np
from epv_utils import is_between


def get_EPV_grid(fname, fpath='grids', as_class=True, origin=None):

    # get complete path for grid file
    if fpath:
        fname = fpath + '//' + fname

    epv = np.loadtxt(fname, delimiter=',')

    # can be returned as ndarray or as an instance of the EPV_Grid class
    if as_class:
        epv = EPV_Grid(grid=epv, dimensions=epv.shape, origin=origin)

    return epv


class EPV_Grid:
    def __init__(self, grid, dimensions=None, origin=None, x_range=None, y_range=None):

        self.grid = grid
        if dimensions is None:
            self.dimensions = grid.shape
            print('Dimensions were not supplied and hence derived from the shape of the grid!')
        else:
            self.dimensions = dimensions
        self.origin = origin
        self.x_range = x_range
        self.y_range = y_range

    def __str__(self):
        if self.origin:
            return f'EPV grid from {self.origin} of {self.dimensions} dimensions'
        else:
            return f'EPV grid of {self.dimensions} dimensions'

    def get_EPV_at_location(self, location, x_range, y_range, attacking_direction='ltr'):

        # make sure correct attacking direcition is given
        assert attacking_direction in ('ltr', 'rtl'), 'attacking_direction needs to be either "ltr" or "rtl"'
        # functions field dimensions overwrite classed field dimensions!
        if x_range and y_range:
            pass
        else:
            if self.x_range and self.y_range:
                x_range = self.x_range
                y_range = self.y_range
            else:
                return ValueError('No field dimensions have been defined in the class object. Therefore you need to '
                                  'specify them in the function!')

        x, y = location

        # Check if position is off the field
        if not is_between(x_range[0], x, x_range[1]) or not is_between(y_range[0], y, y_range[1]):
            return 0.0
        else:
            # rotate grid to account for playing direction of analyzed team
            if attacking_direction == 'rtl':
                grid = np.flip(self.grid)
            else:
                grid = self.grid
            ny, nx = grid.shape  # number of grid cells for both axes
            # dimensions of cells
            dx = abs(x_range[0] - x_range[1]) / float(nx)
            dy = abs(y_range[0] - y_range[1]) / float(ny)
            cx = abs(x_range[0] - x) / abs(x_range[0] - x_range[1]) * nx
            cy = abs(y_range[0] - y) / abs(y_range[0] - y_range[1]) * ny
            return (cy, cx), grid[int(cy), int(cx)]



