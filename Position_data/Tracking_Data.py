import pandas as pd
import numpy as np
from Pitch.My_Pitch import \
    myPitch  # might need adaptation of path depending on whether it is used in pycharm or jupyter notebook
from mplsoccer import Pitch
import matplotlib.pyplot as plt


class tracking_data:

    def __init__(self, data, data_source, x_range_data=None, y_range_data=None,
                 x_range_pitch=None, y_range_pitch=None, mirror_away=None,
                 x_cols_pattern=None, y_cols_pattern=None, scale_to_pitch='mplsoccer',
                 mirror_second_half=None, home=None, away=None, period_col=None):

        self.supported_data_sources = ['metrica']

        # selfs independent of data source and pitch type
        self.org_data = data
        self.data_source = data_source
        self.scale_to_pitch = scale_to_pitch
        self.home = home
        self.away = away

        # selfs preparing for conditional assignment
        self.x_range_data = x_range_data
        self.y_range_data = y_range_data
        self.x_range_pitch = x_range_pitch
        self.y_range_pitch = y_range_pitch
        self.mirror_away = mirror_away
        self.x_cols_pattern = x_cols_pattern
        self.y_cols_pattern = y_cols_pattern
        self.mirror_second_half = mirror_second_half
        self.period_column = period_col

        # if nothing else is specified, we assume the standard values of metrica if metrica is specified
        if self.data_source == 'metrica':
            # standard coordinates of metrica tracking data
            if self.x_range_data is None:
                self.x_range_data = (0, 1)
            if self.y_range_data is None:
                self.y_range_data = (1, 0)
            if self.mirror_second_half is None:
                self.mirror_second_half = True
            if self.mirror_away is None:
                self.mirror_away = []

            # standard naming in metrica data
            if self.x_cols_pattern is None:
                self.x_cols_pattern = 'x'
            if self.y_cols_pattern is None:
                self.y_cols_pattern = 'y'
            if self.period_column is None:
                self.period_column = 'Period_x'
        else:
            raise ValueError(f'You entered {self.data_source}. '
                             f'Unfortunately only {self.supported_data_sources} is/are supported at this stage.')

        # get the intended range for the coordinates based on selected pitch type
        self.supported_pitch_types = ['mplsoccer', 'myPitch']
        if self.scale_to_pitch == 'mplsoccer':
            if self.x_range_pitch is None:
                self.x_range_pitch = (0, 120)
            if self.y_range_pitch is None:
                self.y_range_pitch = (80, 0)
        elif self.scale_to_pitch == 'myPitch':
            if self.x_range_pitch is None:
                self.x_range_pitch = (0, 105)
            if self.y_range_pitch is None:
                self.y_range_pitch = (0, 65)
        elif (self.x_range_pitch is None) or (self.y_range_pitch is None):
            raise ValueError(f'You have not selected a pitch type to which the data is supposed to be scaled.'
                             f'Neither did you supply custom ranges via "x_range_pitch" and "y_range_pitch"'
                             f'Either supply one of {self.supported_pitch_types} to "scale_to_pitch" or '
                             f'Supply tuples of data ranges to "x_range_pitch" and "y_range_pitch".')

        self.data = self.rescale_tracking_data()

    def __str__(self):
        return (
            f"tracking_data object of {self.data_source} of shape {self.data.shape}."
        )

    def rescale_tracking_data(self):

        # deep copy to avoid changes on original data if not intended
        data = self.org_data.copy(deep=True)

        # dimensions dictionary for convenience
        self.dimensions = {self.y_cols_pattern: {
            'data': self.y_range_data,
            'pitch': self.y_range_pitch},
            self.x_cols_pattern: {
                'data': self.x_range_data,
                'pitch': self.x_range_pitch}
        }
        # add the columns home and away and x and y
        self.dimensions[self.x_cols_pattern]['home_columns'] = [c for c in data.columns if
                                                           c[-1].lower() == self.x_cols_pattern and c.startswith(
                                                               'Home')]
        self.dimensions[self.x_cols_pattern]['away_columns'] = [c for c in data.columns if
                                                           c[-1].lower() == self.x_cols_pattern and c.startswith(
                                                               'Away')]
        self.dimensions[self.y_cols_pattern]['home_columns'] = [c for c in data.columns if
                                                           c[-1].lower() == self.y_cols_pattern and c.startswith(
                                                               'Home')]
        self.dimensions[self.y_cols_pattern]['away_columns'] = [c for c in data.columns if
                                                           c[-1].lower() == self.y_cols_pattern and c.startswith(
                                                               'Away')]

        for dim in self.dimensions.keys():
            datamin = self.dimensions[dim]['data'][0]
            datamax = self.dimensions[dim]['data'][1]
            delta_data = datamax - datamin
            self.dimensions[dim]['delta_data'] = delta_data
            pitchmin = self.dimensions[dim]['pitch'][0]
            pitchmax =  self.dimensions[dim]['pitch'][1]
            delta_pitch = pitchmax - pitchmin
            self.dimensions[dim]['delta_pitch'] = delta_pitch
            self.dimensions[dim]['scaling_factor'] = delta_pitch / delta_data
        # print(self.dimensions)

        # for both x and y (or whatever they are called)
        for dim in self.dimensions.keys():
            # if data is oriented in the correct way
            if self.dimensions[dim]['delta_data'] > 0:
                # home
                data[self.dimensions[dim]['home_columns']] = self.dimensions[dim]['pitch'][0] + data[
                    self.dimensions[dim]['home_columns']] * self.dimensions[dim]['scaling_factor']
                # away (mirror away?)
                if dim in self.mirror_away:
                    data[self.dimensions[dim]['away_columns']] = self.dimensions[dim]['pitch'][1] - data[
                        self.dimensions[dim]['away_columns']] * self.dimensions[dim]['scaling_factor']
                else:
                    data[self.dimensions[dim]['away_columns']] = self.dimensions[dim]['pitch'][0] + data[
                        self.dimensions[dim]['away_columns']] * self.dimensions[dim]['scaling_factor']
            # else if data is not oriented in the correct way
            elif self.dimensions[dim]['delta_data'] < 0:
                # home
                data[self.dimensions[dim]['home_columns']] = self.dimensions[dim]['pitch'][1] + data[
                    self.dimensions[dim]['home_columns']] * self.dimensions[dim]['scaling_factor']
                # away (mirror away?)
                if dim in self.mirror_away:
                    data[self.dimensions[dim]['away_columns']] = self.dimensions[dim]['pitch'][0] - data[
                        self.dimensions[dim]['away_columns']] * self.dimensions[dim]['scaling_factor']
                else:
                    data[self.dimensions[dim]['away_columns']] = self.dimensions[dim]['pitch'][1] + data[
                        self.dimensions[dim]['away_columns']] * self.dimensions[dim]['scaling_factor']

        if self.mirror_second_half:
            half_filter = data[self.period_column] == 2
            # home
            data.loc[half_filter, self.dimensions[dim]['home_columns']] = max(self.dimensions[dim]['pitch']) - data[self.dimensions[dim]['home_columns']][half_filter]
            # away
            data.loc[half_filter, self.dimensions[dim]['away_columns']] = max(self.dimensions[dim]['pitch']) - data[self.dimensions[dim]['away_columns']][half_filter]

        return data

    # ------------------------------------------
    # function to plot players for a given frame
    # ------------------------------------------

    def plot_players(self, frame, pitch_col='#1c380e', line_col='white', colors=['red', 'blue']):
        if self.scale_to_pitch == 'mplsoccer':
            pitch = Pitch(pitch_color=pitch_col, line_color=line_col)
            fig, ax = plt.subplots()
            fig.set_facecolor(pitch_col)
            pitch.draw(ax=ax)
        elif self.scale_to_pitch == 'myPitch':
            pitch = myPitch(grasscol=pitch_col)
            fig, ax = plt.subplots()  # figsize=(13.5, 8)
            fig.set_facecolor(pitch_col)
            pitch.plot_pitch(ax=ax)
        else:
            raise ValueError(f'Unfortunately the pitch {self.scale_to_pitch} is not yet supported by this function!')

        # get the row / frame
        plot_data = self.data.iloc[frame]

        # for both teams
        for team, color in zip(['home', 'away'], colors):
            # get x and y values
            x_values = plot_data[self.dimensions[self.x_cols_pattern][''.join([team, '_columns'])]]
            y_values = plot_data[self.dimensions[self.y_cols_pattern][''.join([team, '_columns'])]]
            ax.scatter(x=x_values, y=y_values, s=20, c=color)
        return fig