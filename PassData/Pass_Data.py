import logging
import sys

sys.path.append('C:\\Users\\DavidB\\PycharmProjects\\My_Football_Analytics')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from Basics.Pitch.My_Pitch import myPitch  # might need adaptation of path depending on whether it is used in pycharm
# or jupyter notebook
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import numpy as np
import re
import warnings
import logging


# ------------------------------------------------------------------------
# pass data as its own class with functions to rescale and create shot map
# ------------------------------------------------------------------------

class pass_data:

    def __init__(self, data, data_source=None, x_range_data=None, y_range_data=None, team_col='team',
                 location_col=None, end_location_key=None, pass_col=None, player_col=None, teams=None,
                 scale_to_pitch='myPitch', x_range_pitch=None, y_range_pitch=None, rel_eve_col=None,
                 mirror_away=['x', 'y'], type_key=None, outcome_key=None, minute_col=None, second_col=None,
                 shot_ass_key=None, goal_ass_key=None, cross_key=None, cutback_key=None, switch_key=None,
                 play_pattern_col=None, half_col=None, receiver_key=None):

        supported_data_sources = ['Statsbomb']

        # We need these attributes later independent of data source!
        self.org_data = data
        self.data_source = data_source
        self.x_range_data = x_range_data
        self.y_range_data = y_range_data
        self.team_column = team_col
        self.scale_to_pitch = scale_to_pitch
        self.x_range_pitch = x_range_pitch
        self.y_range_pitch = y_range_pitch
        self.location_column = location_col
        self.end_location_key = end_location_key
        self.pass_column = pass_col
        self.player_column = player_col
        self.minute_column = minute_col
        self.second_column = second_col
        self.mirror_away = mirror_away
        self.type_key = type_key
        self.outcome_key = outcome_key
        self.rel_eve_column = rel_eve_col
        self.shot_ass_key = shot_ass_key
        self.goal_ass_key = goal_ass_key
        self.cross_key = cross_key
        self.cutback_key = cutback_key
        self.switch_key = switch_key
        self.play_pattern_column = play_pattern_col
        self.half_column = half_col
        self.receiver_key = receiver_key

        # usually in Statsbomb both teams play both halves left to right (0 to 120) and there are certain naming
        # conventions. These will be used unless something else is explicitly specified
        if self.data_source == 'Statsbomb':
            if self.location_column is None:
                self.location_column = 'location'
            if self.end_location_key is None:
                self.end_location_key = 'end_location'
            if self.team_column is None:
                self.team_column = 'team'
            if self.player_column is None:
                self.player_column = 'player'
            if self.pass_column is None:
                self.pass_column = 'pass'
            if self.minute_column is None:
                self.minute_column = 'minute'
            if self.second_column is None:
                self.second_column = 'second'
            if self.type_key is None:
                self.type_key = 'type'
            if self.outcome_key is None:
                self.outcome_key = 'outcome'
            if self.rel_eve_column is None:
                self.rel_eve_column = 'related_events'
            if self.play_pattern_column is None:
                self.play_pattern_column = 'play_pattern'
            if self.half_column is None:
                self.half_column = 'period'
            if self.shot_ass_key is None:
                self.shot_ass_key = 'shot_assist'
            if self.goal_ass_key is None:
                self.goal_ass_key = 'goal_assist'
            if self.cross_key is None:
                self.cross_key = 'cross'
            if self.switch_key is None:
                self.switch_key = 'switch'
            if self.cutback_key is None:
                self.cutback_key = 'cut-back'
            if self.receiver_key is None:
                self.receiver_key = 'recipient'
            if self.x_range_data is None:
                self.x_range_data = (0, 120)
            if self.y_range_data is None:
                self.y_range_data = (80, 0)

        elif self.x_range_data is None or self.y_range_data is None:
            raise ValueError(f'You have not selected a data source which which would indicate an original scale.'
                             f'Neither did you supply custom ranges via "x_range_data" and "y_range_data"'
                             f'Either supply one of {self.supported_data_sources} to "data_source" or '
                             f'Supply tuples of data ranges to "x_range_data" and "x_range_data".')

        if teams is None:  # if we do not supply home and away it will try to guess by order (Works for understat)
            self.home_team = data[self.team_column].unique()[0]
            self.away_team = data[self.team_column].unique()[1]
        else:
            self.home_team = teams[0]
            self.away_team = teams[1]

        self.filter1 = data[self.team_column] == self.home_team
        self.filter2 = data[self.team_column] == self.away_team

        # get the intended range for the coordinates based on selected pitch type
        self.supported_pitch_types = ['mplsoccer', 'myPitch']
        if self.scale_to_pitch == 'mplsoccer':
            if self.x_range_pitch or self.y_range_pitch:
                logging.warning("mplsoccer pitch does not allow for a rescaling of the pitch. Axis ranges remain as"
                                "(0, 120) for x and (80, 0) for y!")
            self.x_range_pitch = (0, 120)
            self.y_range_pitch = (80, 0)

        elif self.scale_to_pitch == 'myPitch':
            if self.x_range_pitch is None:
                self.x_range_pitch = (0, 105)
            if self.y_range_pitch is None:
                self.y_range_pitch = (0, 68)
        elif (self.x_range_pitch is None) or (self.y_range_pitch is None):
            raise ValueError(f'You have not selected a pitch type to which the data is supposed to be scaled.'
                             f'Neither did you supply custom ranges via "x_range_pitch" and "y_range_pitch"'
                             f'Either supply one of {self.supported_pitch_types} to "scale_to_pitch" or '
                             f'Supply tuples of data ranges to "x_range_pitch" and "y_range_pitch".')

        # on initializing the data is rescaled, but I can always initialize again based on org_data!
        self.data = self.rescale_pass_data()

    def __str__(self):
        return (
            f"pass_data object of {self.data_source} of shape {self.data.shape}. Coordinate ranges are"
            f" {self.x_range_pitch} for x and {self.y_range_pitch} for y"
        )

    # --------------------------------------------------
    # function to rescale data to a data range of choice
    # --------------------------------------------------
    def rescale_pass_data(self):

        # deep copy to avoid changes on original data if not intended
        data = self.org_data.copy(deep=True)

        if self.x_range_data is None or self.y_range_data is None or self.x_range_pitch is None or self.y_range_pitch is None:
            raise ValueError(f'Oops! Something went wrong. The coordinates for rescaling are missing.')

        # dimensions dictionary for convenience
        self.dimensions = {'y': {
            'data': self.y_range_data,
            'pitch': self.y_range_pitch},
            'x': {
                'data': self.x_range_data,
                'pitch': self.x_range_pitch}
        }
        for dim in self.dimensions.keys():
            datamin = self.dimensions[dim]['data'][0]
            datamax = self.dimensions[dim]['data'][1]
            delta_data = datamax - datamin
            self.dimensions[dim]['delta_data'] = delta_data
            pitchmin = self.dimensions[dim]['pitch'][0]
            pitchmax = self.dimensions[dim]['pitch'][1]
            delta_pitch = pitchmax - pitchmin
            self.dimensions[dim]['delta_pitch'] = delta_pitch
            self.dimensions[dim]['scaling_factor'] = delta_pitch / delta_data

        if self.data_source == 'Statsbomb':
            # collect relevant columns
            player = data[self.player_column]
            team = data[self.team_column]
            minute = data[self.minute_column]
            second = data[self.second_column]
            period = data[self.half_column]
            related_events = data[self.rel_eve_column]
            play_pattern = data[self.play_pattern_column]
            outcome = []
            type = []
            cross = []
            cutback = []
            switch = []
            shot_assist = []
            goal_assist = []
            receiver = []

            outcome += [
                data[self.pass_column][p][self.outcome_key]['name'] if self.outcome_key in data[self.pass_column][
                    p] else "Complete" for p in range(len(data))]
            type += [data[self.pass_column][p][self.type_key]['name'] if self.type_key in data[self.pass_column][
                p] else "Regular" for p in range(len(data))]
            cross += [
                data[self.pass_column][p][self.cross_key] if self.cross_key in data[self.pass_column][p] else False for
                p in range(len(data))]
            cutback += [
                data[self.pass_column][p][self.cutback_key] if self.cutback_key in data[self.pass_column][p] else False
                for p in range(len(data))]
            switch += [
                data[self.pass_column][p][self.switch_key] if self.switch_key in data[self.pass_column][p] else False
                for p in range(len(data))]
            shot_assist += [data[self.pass_column][p][self.shot_ass_key] if self.shot_ass_key in data[self.pass_column][
                p] else False for p in range(len(data))]
            goal_assist += [data[self.pass_column][p][self.goal_ass_key] if self.goal_ass_key in data[self.pass_column][
                p] else False for p in range(len(data))]
            receiver += [
                data[self.pass_column][p][self.receiver_key]['name'] if self.receiver_key in data[self.pass_column][
                    p] else False for p in range(len(data))]

            # start and end location split into separate lists for both x and y
            # loop over pass location and access both x and y
            x1_org = [float(p[0]) for p in data[self.location_column]]
            y1_org = [float(p[1]) for p in data[self.location_column]]
            # access the dictionary stored at column 'pass' and get end_location by key
            x2_org = [float(p[self.end_location_key][0]) for p in data[self.pass_column]]
            y2_org = [float(p[self.end_location_key][1]) for p in data[self.pass_column]]

            pada = pd.DataFrame(zip(player, period, minute, second, team, outcome, type, x1_org, y1_org, x2_org, y2_org,
                                    cross, cutback, switch, shot_assist, goal_assist, play_pattern, related_events,
                                    receiver),
                                columns=[self.player_column, self.half_column, self.minute_column, self.second_column,
                                         self.team_column, self.outcome_key, self.type_key, 'x_initial', 'y_initial',
                                         'x_received', 'y_received', self.cross_key, self.cutback_key, self.switch_key,
                                         self.shot_ass_key, self.goal_ass_key, self.play_pattern_column,
                                         self.rel_eve_column, self.receiver_key])
            coordinates = ['x_initial', 'y_initial', 'x_received', 'y_received']

            for c in coordinates:
                dim = re.sub(pattern='_.*', repl='', string=c)
                # rescale home team coordinates
                pada.loc[self.filter1, c] = pada.loc[self.filter1, c].apply(
                    lambda x: self.dimensions[dim]['pitch'][0] + (x + self.dimensions[dim]['data'][0] * -1) *
                              self.dimensions[dim]['scaling_factor'])

                # rescale away team and if necessary mirror
                if dim in self.mirror_away:
                    pada.loc[self.filter2, c] = pada.loc[self.filter2, c].apply(
                        lambda x: self.dimensions[dim]['pitch'][1] - (x + self.dimensions[dim]['data'][0] * -1) *
                                  self.dimensions[dim]['scaling_factor'])
                else:
                    pada.loc[self.filter2, c] = pada.loc[self.filter2, c].apply(
                        lambda x: self.dimensions[dim]['pitch'][0] + (x + self.dimensions[dim]['data'][0] * -1) *
                                  self.dimensions[dim]['scaling_factor'])

                data = pada

        else:  # currently "Statsbomb" is the only option included (could add others like Statsbomb)
            raise ValueError(
                f'{self.data_source} not supported. At this point, Statsbomb is the only supported '
                f'data format.')

        return data

    def get_passes(self, get, data=None, receiver_get=False, receiver_count=False):

        # allows to supply data instead of using the shotdata.data
        # therefore we can use function on the outcome of the function
        # (e.g. first get all crosses and then the assists)
        if data is None:
            got_data = self.data.copy(deep=True)
        else:
            got_data = data.copy(deep=True)

        if get in [self.shot_ass_key, self.goal_ass_key, self.cutback_key, self.cross_key, self.switch_key]:
            got_data = got_data.loc[got_data[get] == True]
            n = self.count_returner(got_data, receiver=receiver_count)
            return got_data, n

        elif get in list(got_data[self.receiver_key]) and receiver_get:
            got_data = got_data.loc[got_data[self.receiver_key] == get]
            n = self.count_returner(got_data, receiver=receiver_count)
            return got_data, n

        elif get in list(got_data[self.player_column]):
            got_data = got_data.loc[got_data[self.player_column] == get]
            n = self.count_returner(got_data, receiver=receiver_count)
            return got_data, n

        elif get in [self.home_team, self.away_team]:
            got_data = got_data.loc[got_data[self.team_column] == get]
            n = self.count_returner(got_data, receiver=receiver_count)
            return got_data, n

        elif get in list(got_data[self.outcome_key]):
            got_data = got_data.loc[got_data[self.outcome_key] == get]
            n = self.count_returner(got_data, receiver=receiver_count)
            return got_data, n

        elif get in list(got_data[self.type_key]):
            got_data = got_data.loc[got_data[self.type_key] == get]
            n = self.count_returner(got_data, receiver=receiver_count)
            return got_data, n

        elif get in list(got_data[self.play_pattern_column]):
            got_data = got_data.loc[got_data[self.play_pattern_column] == get]
            n = self.count_returner(got_data, receiver=receiver_count)
            return got_data, n

        elif get in list(got_data[self.half_column]):
            got_data = got_data.loc[got_data[self.half_column] == get]
            n = self.count_returner(got_data, receiver=receiver_count)
            return got_data, n

        else:
            raise ValueError(f'No valid value was supplied to "get". You can supply values/strings that occur in any of'
                             f' the following columns: {self.half_column}, {self.play_pattern_column},'
                             f'{self.type_key}, {self.outcome_key}, {self.team_column}, {self.player_column} or'
                             f' one of the following strings:'
                             f'{self.cutback_key}, {self.switch_key}, {self.cross_key}, {self.shot_ass_key},'
                             f' {self.goal_ass_key}.')

    def pass_map(self, plot_direction_of_play=True, data=None, direction_of_play='ltr', pdop_x=1 / 3, pdop_y=0.1,
                 pdop_l=1 / 3, pitch_col='#1c380e', line_col='white', pdop_o=0.2):

        if self.scale_to_pitch == 'mplsoccer':
            pitch = Pitch(pitch_color=pitch_col, line_color=line_col)
            fig, ax = plt.subplots()
            fig.set_facecolor(pitch_col)
            pitch.draw(ax=ax)
        elif self.scale_to_pitch == 'myPitch':
            pitch = myPitch(grasscol=pitch_col, x_range_pitch=self.x_range_pitch, y_range_pitch=self.y_range_pitch)
            fig, ax = plt.subplots()  # figsize=(13.5, 8)
            fig.set_facecolor(pitch_col)
            pitch.plot_pitch(ax=ax)
        else:
            raise ValueError(f'Unfortunately the pitch {self.scale_to_pitch} is not yet supported by this function!')

        if data is None:
            data = self.data
            warnings.warn('Recommended to use filtered data to avoid a mess!', category=Warning)

        teams = data[self.team_column].unique()
        if len(teams) > 1:
            warnings.warn('Recommended to include only one team to avoid a mess!', category=Warning)
        elif plot_direction_of_play:
            if direction_of_play == 'ltr':
                plt.arrow(x=min(self.x_range_pitch) + max(self.x_range_pitch) * pdop_x,
                          y=max(self.y_range_pitch) * pdop_y,
                          dx=pdop_l * (max(self.x_range_pitch) - min(self.x_range_pitch)), dy=0,
                          width=1, color='white', alpha=pdop_o)
            elif direction_of_play == 'rtl':
                plt.arrow(x=max(self.x_range_pitch) - max(self.x_range_pitch) * pdop_x,
                          y=max(self.y_range_pitch) * pdop_y,
                          dx=pdop_l * (max(self.x_range_pitch) - min(self.x_range_pitch)), dy=0,
                          width=1, color='white', alpha=pdop_o)
            else:
                raise Warning('No valid direction of play was supplied. Either specify a direction of play '
                              '["ltr", "rtl"] or disable the plot_direction_of_play.')

        for p in range(len(data['x_initial'])):
            if data[self.outcome_key].iloc[p] == 'Complete':
                plt.arrow(x=data['x_initial'].iloc[p], y=data['y_initial'].iloc[p],
                          dx=data['x_received'].iloc[p] - data['x_initial'].iloc[p],
                          dy=data['y_received'].iloc[p] - data['y_initial'].iloc[p],
                          width=0.1, color='green', length_includes_head=True, head_width=0.75)
            else:
                plt.arrow(x=data['x_initial'].iloc[p], y=data['y_initial'].iloc[p],
                          dx=data['x_received'].iloc[p] - data['x_initial'].iloc[p],
                          dy=data['y_received'].iloc[p] - data['y_initial'].iloc[p],
                          width=0.1, color='red', length_includes_head=True, head_width=0.75)

        return fig

    def pass_network(self, pitch_col='#1c380e', line_col='white', colors=None, data=None, pass_min=5,
                     by_receive=False):

        if colors is None:
            colors = ['#d9534f', '#5bc0de', '#5cb85c', '#428bca', '#faa632', '#c7254e', '#843534', '#ff71ce',
                      '#01cdfe', '#05ffa1', '#b967ff', '#fffb96']

        if self.scale_to_pitch == 'mplsoccer':
            pitch = Pitch(pitch_color=pitch_col, line_color=line_col)
            fig, ax = plt.subplots()
            fig.set_facecolor(pitch_col)
            pitch.draw(ax=ax)
        elif self.scale_to_pitch == 'myPitch':
            pitch = myPitch(grasscol=pitch_col, x_range_pitch=self.x_range_pitch, y_range_pitch=self.y_range_pitch)
            fig, ax = plt.subplots()  # figsize=(13.5, 8)
            fig.set_facecolor(pitch_col)
            pitch.plot_pitch(ax=ax)
        else:
            raise ValueError(f'Unfortunately the pitch {self.scale_to_pitch} is not yet supported by this function!')

        if data is None:
            data = self.data
            warnings.warn('Recommended to use filtered data to avoid a mess!', category=Warning)

        # get starting 11
        XI = data[self.player_column].unique()[0:11]

        # create a dictionary with all values needed for network plotting
        network = dict()

        if not by_receive:
            # for each player in starting XI
            for p in XI:
                network[p] = dict()
                pP, nP = self.get_passes(get=p, data=data)  # all passed played by p
                network[p]['x_avg'] = pP['x_initial'].mean()  # average x of pass origin for p
                network[p]['y_avg'] = pP['y_initial'].mean()  # average y of pass origin for p
                network[p]['n'] = nP  # number of passes played by p
                pPs, nPs = self.get_passes(get="Complete", data=pP)  # all completed passes of p
                network[p]['n_complete'] = nPs  # number of completed passes of p
                network[p]['receivers'] = {}
                receivers = list(pP[self.receiver_key].unique())  # all players who received as pass from p
                for r in receivers:
                    if r == p or r not in XI:
                        pass
                    else:
                        pR, nR = self.get_passes(get=r, data=pP, receiver_get=True)  # all passes played from p to r
                        network[p]['receivers'][r] = nR  # number of passes from p to r

            for P, p in enumerate(XI):
                if len(colors) < 11:
                    warnings.warn('Recommended to supply a list of colors with one per player (i.e. 11). '
                                  'If less than 11 colors are supplied, the first one will be used for all players')
                    if type(colors) == str:
                        color = colors
                    else:
                        color = colors[0]
                else:
                    color = colors[P]
                plt.scatter(network[p]['x_avg'], network[p]['y_avg'], s=network[p]['n'] * 3, color=color)
                for r in network[p]['receivers']:
                    if r != 'None' and r in XI:
                        if network[p]['receivers'][r] > pass_min:
                            plt.arrow(x=network[p]['x_avg'], y=network[p]['y_avg'],
                                      dx=network[r]['x_avg'] - network[p]['x_avg'],
                                      dy=network[r]['y_avg'] - network[p]['y_avg'], color=color,
                                      alpha=0.5, width=network[p]['receivers'][r] / 10, length_includes_head=True,
                                      head_width=network[p]['receivers'][r] / 2,
                                      head_length=network[p]['receivers'][r] / 3)
        elif by_receive:
            for r in XI:
                network[r] = dict()
                rR, nR = self.get_passes(get=r, data=data, receiver_get=True,
                                         receiver_count=True)  # all passed received by r
                network[r]['x_avg'] = rR['x_received'].mean()  # average x where r received passed
                network[r]['y_avg'] = rR['y_received'].mean()  # average y where r received passed
                network[r]['n'] = nR  # number of passes received by r
                rRs, nRs = self.get_passes(get="Complete", data=rR, receiver_count=True)  # all completed passes to r
                network[r]['n_complete'] = nRs  # number of completed passes to r
                network[r]['passers'] = {}
                passers = list(rR[self.player_column].unique())  # all players who passed to r
                for p in passers:
                    if p == r or p not in XI:
                        pass
                    else:
                        pP, nP = self.get_passes(get=p, data=rR)  # all passes received by r from p
                        network[r]['passers'][p] = nP  # number of passes received by r from p

            for R, r in enumerate(XI):
                if len(colors) < 11:
                    warnings.warn('Recommended to supply a list of colors with one per player (i.e. 11). '
                                  'If less than 11 colors are supplied, the first one will be used for all players')
                    if type(colors) == str:
                        color = colors
                    else:
                        color = colors[0]
                else:
                    color = colors[R]
                plt.scatter(network[r]['x_avg'], network[r]['y_avg'], s=network[r]['n'] * 3, color=color)
                for p in network[r]['passers']:
                    if p != 'None' and p in XI:
                        if network[r]['passers'][p] > pass_min:
                            plt.arrow(x=network[p]['x_avg'], y=network[p]['y_avg'],
                                      dx=network[r]['x_avg'] - network[p]['x_avg'],
                                      dy=network[r]['y_avg'] - network[p]['y_avg'], color=color,
                                      alpha=0.5, width=network[r]['passers'][p] / 10, length_includes_head=True,
                                      head_width=network[r]['passers'][p] / 2,
                                      head_length=network[r]['passers'][p] / 3)

        return fig, network

    def count_returner(self, data, receiver=False):
        if not receiver:
            if len(list(data[self.team_column].unique())) > 1:  # more than one team?!
                # per team
                n_home = len(data.loc[self.filter1])
                n_away = len(data.loc[self.filter2])
                n_dict = {self.home_team: n_home,
                          self.away_team: n_away}
                return n_dict
            elif len(list(data[self.player_column].unique())) > 1:  # just one team but more than one player?!
                # per player
                players = list(data[self.player_column].unique())
                n_dict = {}
                for player in players:
                    n = len(data.loc[data[self.player_column] == player])  # just one player?!
                    n_dict[player] = n
                return n_dict
            else:
                n = len(data)
                return n
        elif receiver:
            if len(list(data[self.team_column].unique())) > 1:  # more than one team?!
                # per team
                n_home = len(data.loc[self.filter1])
                n_away = len(data.loc[self.filter2])
                n_dict = {self.home_team: n_home,
                          self.away_team: n_away}
                return n_dict
            elif len(list(data[self.receiver_key].unique())) > 1:  # just one team but more than one player?!
                # per player
                players = list(data[self.receiver_key].unique())
                n_dict = {}
                for player in players:
                    n = len(data.loc[data[self.receiver_key] == player])  # just one player?!
                    n_dict[player] = n
                return n_dict
            else:
                n = len(data)
                return n
