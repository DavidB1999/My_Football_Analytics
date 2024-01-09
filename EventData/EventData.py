import sys

sys.path.append('C:\\Users\\DavidB\\PycharmProjects\\My_Football_Analytics')
import pandas as pd
import numpy as np
import logging
from Basics.Pitch.My_Pitch import myPitch
from mplsoccer import Pitch
import matplotlib.pyplot as plt


def load_event_data(datadir, fname):
    file = datadir + '/' + fname
    print(file)
    events = pd.read_csv(file)  # read data
    return events


class event_data:

    def __init__(self, data, data_source, x_range_data=None, y_range_data=None,
                 x_range_pitch=None, y_range_pitch=None, scale_to_pitch='myPitch',
                 mirror_second_half=None, fps=None, colors=['red', 'blue', 'black'],
                 mirror_away=None):

        self.dimensions = None
        self.supported_data_sources = ['metrica']
        self.supported_pitches = ['mplsoccer', 'myPitch']

        self.org_data = data
        self.data_source = data_source
        self.x_range_data = x_range_data
        self.y_range_data = y_range_data
        self.x_range_pitch = x_range_pitch
        self.y_range_pitch = y_range_pitch
        self.pitch = scale_to_pitch
        self.mirror_second_half = mirror_second_half
        self.mirror_away = mirror_away
        self.fps = fps
        self.colors = colors

        # if nothing else is specified, we assume the standard values of metrica if metrica is specified
        if self.data_source == 'metrica':
            self.team_col = 'Team'
            self.type_col = 'Type'
            self.subtype_col = 'Subtype'
            self.outcome_col = None
            self.period_col = 'Period'
            self.start_frame_col = 'Start Frame'
            self.end_frame_col = 'End Frame'
            self.start_time_s_col = 'Start Time [s]'
            self.end_time_s_col = 'End Time [s]'
            self.start_time_col = None
            self.end_time_col = None
            self.player_col = 'From'
            self.receiver_col = 'To'
            self.start_x = 'Start X'
            self.start_y = 'Start Y'
            self.end_x = 'End X'
            self.end_y = 'End Y'

            # standard coordinates of metrica tracking data
            if self.x_range_data is None:
                self.x_range_data = (0, 1)
            if self.y_range_data is None:
                self.y_range_data = (1, 0)

            """ 
            In metrica's event data teams switch playing direction between halves
            depending on our aim we could mirror so that the team plays in the same direction all game long
            just needs to be consistent with the way we handle tracking data if we want to combine
            """
            if self.mirror_second_half is None:
                self.mirror_second_half = True
            if self.mirror_away is None:
                self.mirror_away = []

        else:
            raise ValueError(f'You entered {self.data_source}. '
                             f'Unfortunately only {self.supported_data_sources} is/are supported at this stage.')

        if self.pitch == 'mplsoccer':
            if self.x_range_pitch or self.y_range_pitch:
                logging.warning("mplsoccer pitch does not allow for a rescaling of the pitch. Axis ranges remain as"
                                "(0, 120) for x and (80, 0) for y!")
            self.x_range_pitch = (0, 120)
            self.y_range_pitch = (80, 0)
        elif self.pitch == 'myPitch':
            if self.x_range_pitch is None:
                self.x_range_pitch = (0, 105)
            if self.y_range_pitch is None:
                self.y_range_pitch = (0, 68)
        elif (self.x_range_pitch is None) or (self.y_range_pitch is None):
            raise ValueError(f'You have not selected a pitch type to which the data is supposed to be scaled.'
                             f'Neither did you supply custom ranges via "x_range_pitch" and "y_range_pitch"'
                             f'Either supply one of {self.supported_pitch_types} to "scale_to_pitch" or '
                             f'Supply tuples of data ranges to "x_range_pitch" and "y_range_pitch".')

        self.data = self.rescale_event_data()

    def __str__(self):
        return (
            f'event_data object with data from {self.data_source}'
        )

    def rescale_event_data(self):

        data = self.org_data.copy(deep=True)
        rows = len(data)
        self.dimensions = {'x': {
            'data': self.x_range_data,
            'pitch': self.x_range_pitch},
            'y': {
                'data': self.y_range_data,
                'pitch': self.y_range_pitch}
        }

        data_dict = {
            'Team': data[self.team_col].values if self.team_col else np.repeat(np.nan, rows),
            'Type': data[self.type_col].values if self.type_col else np.repeat(np.nan, rows),
            'Subtype': data[self.subtype_col].values if self.subtype_col else np.repeat(np.nan, rows),
            'Outcome': data[self.outcome_col].values if self.outcome_col else np.repeat(np.nan, rows),
            'Period': data[self.period_col].values if self.period_col else np.repeat(np.nan, rows),
            'Start_Frame': data[self.start_frame_col].values if self.start_frame_col else np.repeat(np.nan, rows),
            'Start_Time_[s]': data[self.start_time_s_col].values if self.start_time_s_col else np.repeat(np.nan, rows),
            'Start_Time': data[self.start_time_col].values if self.start_time_col else np.repeat(np.nan, rows),
            'End_Frame': data[self.end_frame_col].values if self.end_frame_col else np.repeat(np.nan, rows),
            'End_Time_[s]': data[self.end_time_s_col].values if self.end_time_s_col else np.repeat(np.nan, rows),
            'End_Time': data[self.end_time_col].values if self.end_time_col else np.repeat(np.nan, rows),
            'Player': data[self.player_col].values if self.player_col else np.repeat(np.nan, rows),
            'Receiver': data[self.receiver_col].values if self.receiver_col else np.repeat(np.nan, rows),
            'Start_x': data[self.start_x].values if self.start_x else np.repeat(np.nan, rows),
            'Start_y': data[self.start_y].values if self.start_y else np.repeat(np.nan, rows),
            'End_x': data[self.end_x].values if self.end_x else np.repeat(np.nan, rows),
            'End_y': data[self.end_y].values if self.end_y else np.repeat(np.nan, rows),
        }

        new_data = pd.DataFrame(data_dict)
        new_data.dropna(axis='columns', how='all', inplace=True)

        for dim in self.dimensions:
            self.dimensions[dim]['delta_data'] = self.dimensions[dim]['data'][1] - self.dimensions[dim]['data'][0]
            self.dimensions[dim]['delta_pitch'] = self.dimensions[dim]['pitch'][1] - self.dimensions[dim]['pitch'][0]
            self.dimensions[dim]['scaling_factor'] = self.dimensions[dim]['delta_pitch'] / self.dimensions[dim][
                'delta_data']

        new_data['Start_x'] = self.dimensions['x']['pitch'][0] + (
                new_data['Start_x'] + self.dimensions['x']['data'][0] * (-1)) * self.dimensions['x'][
                                  'scaling_factor']
        new_data['End_x'] = self.dimensions['x']['pitch'][0] + (
                new_data['End_x'] + self.dimensions['x']['data'][0] * (-1)) * self.dimensions['x']['scaling_factor']
        new_data['Start_y'] = self.dimensions['y']['pitch'][0] + (
                new_data['Start_y'] + self.dimensions['y']['data'][0] * (-1)) * self.dimensions['y'][
                                  'scaling_factor']
        new_data['End_y'] = self.dimensions['y']['pitch'][0] + (
                new_data['End_y'] + self.dimensions['y']['data'][0] * (-1)) * self.dimensions['y']['scaling_factor']

        # mirroring?!
        if 'x' in self.mirror_away:
            new_data['Start_x'][new_data['Team'] == 'Away'] = self.dimensions['x']['pitch'][0] - (
                    new_data['Start_x'][new_data['Team'] == 'Away'] + self.dimensions['x']['pitch'][1] * (-1)) * 1
            new_data['End_x'][new_data['Team'] == 'Away'] = self.dimensions['x']['pitch'][0] - (
                    new_data['End_x'][new_data['Team'] == 'Away'] + self.dimensions['x']['pitch'][1] * (-1)) * 1
        if 'y' in self.mirror_away:
            new_data['Start_y'][new_data['Team'] == 'Away'] = self.dimensions['y']['pitch'][0] - (
                    new_data['Start_y'][new_data['Team'] == 'Away'] + self.dimensions['y']['pitch'][1] * (-1)) * 1
            new_data['End_y'][new_data['Team'] == 'Away'] = self.dimensions['y']['pitch'][0] - (
                    new_data['End_y'][new_data['Team'] == 'Away'] + self.dimensions['y']['pitch'][1] * (-1)) * 1

        if self.mirror_second_half:
            half_filter = new_data['Period'] == 2
            new_data.loc[half_filter, 'Start_x'] = self.dimensions['x']['pitch'][0] - (
                    new_data.loc[half_filter, 'Start_x'] + self.dimensions['x']['pitch'][1] * (-1)) * 1
            new_data.loc[half_filter, 'End_x'] = self.dimensions['x']['pitch'][0] - (
                    new_data.loc[half_filter, 'End_x'] + self.dimensions['x']['pitch'][1] * (-1)) * 1
            new_data.loc[half_filter, 'Start_y'] = self.dimensions['y']['pitch'][0] - (
                    new_data.loc[half_filter, 'Start_y'] + self.dimensions['y']['pitch'][1] * (-1)) * 1
            new_data.loc[half_filter, 'End_y'] = self.dimensions['y']['pitch'][0] - (
                    new_data.loc[half_filter, 'End_y'] + self.dimensions['y']['pitch'][1] * (-1)) * 1

        return new_data

    def get_event_by_type(self, event_type):
        ev_da = self.data

        ev_da = ev_da.loc[ev_da['Type'] == event_type]
        return ev_da
