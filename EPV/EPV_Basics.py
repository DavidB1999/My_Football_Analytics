import sys

sys.path.append('C:\\Users\\DavidB\\PycharmProjects\\My_Football_Analytics')

import numpy as np
from epv_utils import is_between
from Basics.Pitch.My_Pitch import myPitch
from mplsoccer import Pitch
import matplotlib.pyplot as plt


def get_EPV_grid(fname, fpath='grids', as_class=True, origin=None, td_object=None, team='home'):
    # get complete path for grid file
    if fpath:
        fname = fpath + '//' + fname

    epv = np.loadtxt(fname, delimiter=',')

    # can be returned as ndarray or as an instance of the EPV_Grid class
    if as_class:
        epv = EPV_Grid(grid=epv, grid_dimensions=epv.shape, origin=origin, td_object=td_object, team=team)

    return epv


class EPV_Grid:
    def __init__(self, grid, grid_dimensions=None, origin=None, td_object=None, x_range=None, y_range=None,
                 scale_to_pitch=None, team='home'):

        self.grid = grid
        if grid_dimensions:
            self.grid_dimensions = grid_dimensions  # (ny, nx)
        else:
            self.dimensions = grid.shape
            print('Dimensions were not supplied and hence derived from the shape of the grid!')
        self.origin = origin
        self.td_object = td_object
        if self.td_object:
            self.x_range_grid = self.td_object.x_range_pitch
            self.y_range_grid = self.td_object.y_range_pitch
            self.scale_to_pitch = self.td_object.scale_to_pitch
        elif x_range and y_range and scale_to_pitch:
            self.x_range_grid = x_range
            self.y_range_grid = y_range
            self.scale_to_pitch = scale_to_pitch
        else:
            raise ValueError('You need to either pass a td_object with defined dimensions (preferred) or pass x_range '
                             'and y_range as well as pitch type manually.')
        self.team = team
        if self.td_object:
            if self.team == 'home' or self.team == 'Home':
                self.playing_direction = self.td_object.playing_direction_home
            elif self.team == 'away' or self.team == 'Away':
                self.playing_direction = self.td_object.playing_direction_away
            else:
                raise ValueError('teams should be either "Home" or "Away" as defined in td_object!')

    def __str__(self):
        if self.origin:
            return f'EPV grid from {self.origin} of {self.grid_dimensions} dimensions'
        else:
            return f'EPV grid of {self.grid_dimensions} dimensions'

    def get_EPV_at_location(self, location):

        # make sure correct attacking direcition is given
        assert self.playing_direction in ('ltr', 'rtl'), 'attacking_direction needs to be either "ltr" or "rtl"'
        # functions field dimensions overwrite classed field dimensions!
        x, y = location

        # Check if position is off the field
        if not is_between(self.x_range_grid[0], x, self.x_range_grid[1]) or not is_between(self.y_range_grid[0], y,
                                                                                           self.y_range_grid[1]):
            return 0.0
        else:
            # rotate grid to account for playing direction of analyzed team
            if self.playing_direction == 'rtl':
                grid = np.flip(self.grid)
            else:
                grid = self.grid
            ny, nx = grid.shape  # number of grid cells for both axes
            # dimensions of cells
            dx = abs(self.x_range_grid[0] - self.x_range_grid[1]) / float(nx)
            dy = abs(self.y_range_grid[0] - self.y_range_grid[1]) / float(ny)
            cx = abs(self.x_range_grid[0] - x) / abs(self.x_range_grid[0] - self.x_range_grid[1]) * nx - 0.0001
            cy = abs(self.y_range_grid[0] - y) / abs(self.y_range_grid[0] - self.y_range_grid[1]) * ny - 0.0001
            return (cy, cx), grid[int(cy), int(cx)]

    def plot_grid(self, pitch_col='white', line_col='#444444'):

        # rotate grid to account for playing direction of analyzed team
        if self.playing_direction == 'rtl':
            grid = np.flip(self.grid)
        else:
            grid = self.grid
        # plot pitch
        if self.scale_to_pitch == 'mplsoccer':
            pitch = Pitch(pitch_color=pitch_col, line_color=line_col)
            fig, ax = plt.subplots()
            fig.set_facecolor(pitch_col)
            pitch.draw(ax=ax)
        elif self.scale_to_pitch == 'myPitch':
            pitch = myPitch(grasscol=pitch_col, linecol=line_col, x_range_pitch=self.x_range_grid,
                            y_range_pitch=self.y_range_grid)
            fig, ax = plt.subplots()  # figsize=(13.5, 8)
            fig.set_facecolor(pitch_col)
            pitch.plot_pitch(ax=ax)

        ax.imshow(grid, extent=(self.x_range_grid[0], self.x_range_grid[1], self.y_range_grid[0],
                                self.y_range_grid[1]),
                  vmin=0.0, vmax=0.6, cmap='Greens', alpha=0.6)

        return fig, ax
