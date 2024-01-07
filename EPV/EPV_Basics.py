import sys

sys.path.append('C:\\Users\\DavidB\\PycharmProjects\\My_Football_Analytics')

import numpy as np
from epv_utils import is_between
from Basics.Pitch.My_Pitch import myPitch
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import torch
import Position_data.PitchControl.pitch_control as pc


def get_EPV_grid(fname, fpath='grids', as_class=True, origin=None, td_object=None, team='Home'):
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
                 scale_to_pitch=None, team='Home', cmaps=['Reds', 'Blues', 'bwr_r', 'bwr']):

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
            if self.team == 'Home':
                self.playing_direction = self.td_object.playing_direction_home
                self.cmap = cmaps[0]
            elif self.team == 'Away':
                self.playing_direction = self.td_object.playing_direction_away
                self.cmap = cmaps[1]
            else:
                raise ValueError('teams should be either "Home" or "Away" as defined in td_object!')

        self.AV_grid = None
        self.cmaps = cmaps

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

    def plot_grid(self, pitch_col='white', line_col='#444444', cmap=None):

        if cmap:
            pass
        else:
            cmap=self.cmap
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
                  vmin=0.0, vmax=0.6, cmap=cmap, alpha=0.6)

        return fig, ax

    def get_AV_grid(self, frame, pc_version='Spearman', pc_implementation='GL', pc_reaction_time=0.7,
                    pc_max_player_speed=None, pc_average_ball_speed=15, pc_sigma=0.45, pc_lambda=4.3, pc_device='cpu',
                    pc_first_frame_calc=0, pc_last_frame_calc=250, pc_batch_size=250, pc_reference='x',
                    pc_assumed_reference_x=105, pc_assumed_reference_y=68):

        pitch_control_grid = pc.tensor_pitch_control(td_object=self.td_object, version=pc_version,
                                                     implementation=pc_implementation, jitter=1e-12,
                                                     pos_nan_to=-1000, vel_nan_to=0, remove_first_frames=0,
                                                     reaction_time=pc_reaction_time,
                                                     max_player_speed=pc_max_player_speed,
                                                     average_ball_speed=pc_average_ball_speed, sigma=pc_sigma,
                                                     lamb=pc_lambda, n_grid_points_x=self.grid_dimensions[0],
                                                     n_grid_points_y=self.grid_dimensions[1], device=pc_device,
                                                     dtype=torch.float32, first_frame=pc_first_frame_calc,
                                                     last_frame=pc_last_frame_calc, batch_size=pc_batch_size,
                                                     deg=50, max_int=500, team=self.team, return_pcpp=False,
                                                     fix_tti=True, reference=pc_reference,
                                                     assumed_reference_x=pc_assumed_reference_x,
                                                     assumed_reference_y=pc_assumed_reference_y)

        # if Fernandez we need to adapt dimensions of pc tensor
        if pc_version == 'Fernandez':
            pitch_control_grid = pitch_control_grid.reshape(pitch_control_grid.shape[0], self.grid_dimensions[0],
                                                            self.grid_dimensions[1])
        pitch_control_grid = pitch_control_grid[frame - pc_first_frame_calc]
        AV_grid = pitch_control_grid * self.grid

        return AV_grid

    def plot_AV_grid(self, frame, AV_grid=None, pc_version='Spearman', pc_implementation='GL', pc_reaction_time=0.7,
                     pc_max_player_speed=None, pc_average_ball_speed=15, pc_sigma=0.45, pc_lambda=4.3, pc_device='cpu',
                     pc_first_frame_calc=0, pc_last_frame_calc=250, pc_batch_size=250, pc_reference='x',
                     pc_assumed_reference_x=105, pc_assumed_reference_y=68, cmap=None):

        frame_number = frame - pc_first_frame_calc
        if cmap:
            pass
        else:
            cmap = self.cmap

        # plot players
        if AV_grid:
            pass
        else:
            pitch_control_grid = pc.tensor_pitch_control(td_object=self.td_object, version=pc_version,
                                                         implementation=pc_implementation, jitter=1e-12,
                                                         pos_nan_to=-1000, vel_nan_to=0, remove_first_frames=0,
                                                         reaction_time=pc_reaction_time,
                                                         max_player_speed=pc_max_player_speed,
                                                         average_ball_speed=pc_average_ball_speed, sigma=pc_sigma,
                                                         lamb=pc_lambda, n_grid_points_x=self.grid_dimensions[0],
                                                         n_grid_points_y=self.grid_dimensions[1], device=pc_device,
                                                         dtype=torch.float32, first_frame=pc_first_frame_calc,
                                                         last_frame=pc_last_frame_calc, batch_size=pc_batch_size,
                                                         deg=50, max_int=500, team=self.team, return_pcpp=False,
                                                         fix_tti=True, reference=pc_reference,
                                                         assumed_reference_x=pc_assumed_reference_x,
                                                         assumed_reference_y=pc_assumed_reference_y)
        # if Fernandez we need to adapt dimensions of pc tensor
        if pc_version == 'Fernandez':
            pitch_control_grid = pitch_control_grid.reshape(pitch_control_grid.shape[0], self.grid_dimensions[0],
                                                            self.grid_dimensions[1])
        pitch_control_grid = pitch_control_grid[frame_number]
        print(pitch_control_grid.shape)
        print(self.grid.shape)
        AV_grid = pitch_control_grid * self.grid

        fig, ax = self.td_object.plot_players(frame=frame_number, velocities=True, pitch_col='white',
                                              line_col='#444444')

        if pc_version == 'Spearman':
            ax.imshow(self.grid, extent=(
                self.td_object.x_range_pitch[0], self.td_object.x_range_pitch[1], self.td_object.y_range_pitch[0],
                self.td_object.y_range_pitch[1]), cmap=cmap, alpha=0.6, vmin=0.0, vmax=0.6, origin='lower')
        elif pc_version == 'Fernandez':
            ax.imshow(AV_grid, extent=(
                self.td_object.x_range_pitch[0], self.td_object.x_range_pitch[1], self.td_object.y_range_pitch[0],
                self.td_object.y_range_pitch[1]), cmap=cmap, alpha=0.6, vmin=0.0, vmax=0.6, origin='lower')
        else:
            raise ValueError(f'{pc_version} is not a valid version. Chose either "Fernandez" or "Spearman"')

        return fig, ax


