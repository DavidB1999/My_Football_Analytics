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
import logging


# --------------------------------------------------------------------------------------
# shot data as its own class with functions to rescale and create shot map and xG-charts
# --------------------------------------------------------------------------------------

class shot_data:
    def __init__(self, data, data_source=None, x_range_data=None, y_range_data=None, team_col='team',
                 x_col=None, y_col=None, xg_col=None, minute_col=None, result_col=None, player_col=None,
                 scale_to_pitch='myPitch', x_range_pitch=None, y_range_pitch=None,
                 mirror_away=['x', 'y'], location_col=None, shot_col=None, xg_key=None, end_location_key=None,
                 outcome_key=None, teams=None):
        # we need these attributes later independent of data source
        self.org_data = data
        self.data_source = data_source
        self.x_range_data = x_range_data
        self.y_range_data = y_range_data
        self.team_column = team_col
        self.x_col = x_col
        if self.x_col is None:
            self.x_col = 'x'
        self.y_col = y_col
        if self.y_col is None:
            self.y_col = 'y'
        self.xg_col = xg_col
        if self.xg_col is None:
            self.xg_col = 'xG'
        self.result_col = result_col
        if self.result_col is None:
            self.result_col = 'result'
        self.minute_col = minute_col
        self.player_col = player_col
        self.scale_to_pitch = scale_to_pitch
        self.x_range_pitch = x_range_pitch
        self.y_range_pitch = y_range_pitch
        self.mirror_away = mirror_away
        self.location_column = location_col
        self.shot_column = shot_col
        self.xg_key = xg_key
        self.end_location_key = end_location_key
        self.outcome_key = outcome_key

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

        supported_data_source = ['Understat', 'Statsbomb']

        if self.data_source == 'Understat':
            if self.x_range_data is None:
                self.x_range_data = (0, 1)
            if self.y_range_data is None:
                self.y_range_data = (0, 1)
            if self.minute_col is None:
                self.minute_col = 'minute'
            if self.result_col is None:
                self.result_col = 'result'
            if self.player_col is None:
                self.player_col = 'player'
            if self.team_column is None:
                self.team_column = 'team'
            if self.mirror_away is None:
                self.mirror_away = ['x', 'y']

        elif self.data_source == 'Statsbomb':
            if self.x_range_data is None:
                self.x_range_data = (0, 120)
            if self.y_range_data is None:
                self.y_range_data = (80, 0)
            if self.location_column is None:
                self.location_column = 'location'
            if self.minute_col is None:
                self.minute_col = 'minute'
            if self.team_column is None:
                self.team_column = 'team'
            if self.player_col is None:
                self.player_col = 'player'
            if self.shot_column is None:
                self.shot_column = 'shot'
            if self.outcome_key is None:
                self.outcome_key = 'outcome'
            if self.end_location_key is None:
                self.end_location_key = 'end_location'
            if self.xg_key is None:
                self.xg_key = 'statsbomb_xg'
            if self.mirror_away is None:
                self.mirror_away = ['x', 'y']

        elif self.x_range_data is None or self.y_range_data is None:
            raise ValueError(f'You have not selected a data source which which would indicate an original scale.'
                             f'Neither did you supply custom ranges via "x_range_data" and "y_range_data"'
                             f'Either supply one of {supported_data_source} to "data_source" or '
                             f'Supply tuples of data ranges to "x_range_data" and "x_range_data".')

        if teams is None:  # if we do not supply home and away it will try to guess by order (Works for understat)
            self.home_team = data[self.team_column].unique()[0]
            self.away_team = data[self.team_column].unique()[1]
        else:
            self.home_team = teams[0]
            self.away_team = teams[1]
        self.filter1 = data[self.team_column] == self.home_team
        self.filter2 = data[self.team_column] == self.away_team

        # on initializing the data is rescaled, but I can always initialize again based on org_data!
        self.data = self.rescale_shot_data()

    def __str__(self):
        return (
            f"shot_data object of {self.data_source} of shape {self.data.shape}."
        )

    # --------------------------------------------------
    # function to rescale data to a data range of choice
    # --------------------------------------------------
    def rescale_shot_data(self):

        # deep copy to avoid changes on original data if not intended
        data = self.org_data.copy(deep=True)

        # dimensions dictionary for convenience
        self.dimensions = {self.y_col: {
            'data': self.y_range_data,
            'pitch': self.y_range_pitch},
            self.x_col: {
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

        # print(self.dimensions)

        # this works without data_source being supplied but this nevertheless
        # requires a format similar to understat shot data
        if self.data_source == 'Understat' or (self.data_source is None):

            # convert to float / int
            data[self.x_col] = data[self.x_col].astype(float)
            data[self.y_col] = data[self.y_col].astype(float)
            data[self.xg_col] = data[self.xg_col].astype(float)
            data[self.minute_col] = data[self.minute_col].astype(int)

            # for both the x and y coordinates
            for dim in self.dimensions.keys():
                # rescale home team coordinates
                data.loc[self.filter1, dim] = data.loc[self.filter1, dim].apply(
                    lambda x: self.dimensions[dim]['pitch'][0] + (x + self.dimensions[dim]['data'][0] * -1) *
                              self.dimensions[dim]['scaling_factor'])

                # rescale away team and if necessary mirror
                if dim in self.mirror_away:
                    data.loc[self.filter2, dim] = data.loc[self.filter2, dim].apply(
                        lambda x: self.dimensions[dim]['pitch'][1] - (x + self.dimensions[dim]['data'][0] * -1) *
                                  self.dimensions[dim]['scaling_factor'])
                else:
                    data.loc[self.filter2, dim] = data.loc[self.filter2, dim].apply(
                        lambda x: self.dimensions[dim]['pitch'][0] + (x + self.dimensions[dim]['data'][0] * -1) *
                                  self.dimensions[dim]['scaling_factor'])


        elif self.data_source == 'Statsbomb':
            # collect relevant columns
            player = data[self.player_col]
            minute = data[self.minute_col]
            team = data[self.team_column]
            result = []
            xG = []

            result = [data[self.shot_column][s][self.outcome_key]['name'] for s in range(len(data))]
            xG = [float(data[self.shot_column][s][self.xg_key]) for s in range(len(data))]

            # location split into separate lists for both x and y
            # loop over pass location and access both x and y
            x_org = []
            y_org = []

            for p in data[self.location_column]:
                x_org.append(float(p[0]))
                y_org.append(float(p[1]))

            sd = pd.DataFrame(zip(player, minute, team, result, x_org, y_org, xG),
                              columns=[self.player_col, self.minute_col, self.team_column, self.result_col,
                                       self.x_col, self.y_col, self.xg_col])

            # for both the x and y coordinates
            for dim in self.dimensions.keys():
                # rescale home team coordinates
                sd.loc[self.filter1, dim] = sd.loc[self.filter1, dim].apply(
                    lambda x: self.dimensions[dim]['pitch'][0] + (x + self.dimensions[dim]['data'][0] * -1) *
                              self.dimensions[dim]['scaling_factor'])

                # rescale away team and if necessary mirror
                if dim in self.mirror_away:
                    sd.loc[self.filter2, dim] = sd.loc[self.filter2, dim].apply(
                        lambda x: self.dimensions[dim]['pitch'][1] - (x + self.dimensions[dim]['data'][0] * -1) *
                                  self.dimensions[dim]['scaling_factor'])
                else:
                    sd.loc[self.filter2, dim] = sd.loc[self.filter2, dim].apply(
                        lambda x: self.dimensions[dim]['pitch'][0] + (x + self.dimensions[dim]['data'][0] * -1) *
                                  self.dimensions[dim]['scaling_factor'])

            data = sd

        else:  # currently "Understat" and "Statsbomb" are the only options included
            raise ValueError(
                f'{self.data_source} not supported. At this point, Statsbomb and Understat are the only supported '
                f'data formats.')

        return data

    # -------------------------------------
    # function to get goal count for a team
    # -------------------------------------

    def count_goals(self, team):
        if team == 'home':
            ngoals = len(self.data[self.result_col][self.data[self.result_col] == 'Goal'][
                             self.data[self.team_column] == self.home_team])
            + len(self.data[self.result_col][self.data[self.result_col] == 'OwnGoal'][
                      self.data[self.team_column] == self.away_team])
        elif team == 'away':
            ngoals = len(self.data[self.result_col][self.data[self.result_col] == 'Goal'][
                             self.data[self.team_column] == self.away_team])
            + len(self.data[self.result_col][self.data[self.result_col] == 'OwnGoal'][
                      self.data[self.team_column] == self.home_team])
        else:
            raise ValueError(f'You need to supply either "home" or "away" to the team parameter '
                             f'but supplied {team}!')
        return ngoals

    # -------------------------------------
    # function to get xG count for a team
    # -------------------------------------
    def xG_score(self, team):
        if team == 'home':
            xG_score = sum(self.data.loc[self.filter1, self.xg_col])
        elif team == 'away':
            xG_score = sum(self.data.loc[self.filter2, self.xg_col])
        else:
            raise ValueError(f'You need to supply either "home" or "away" to the team parameter '
                             f'but supplied {team}!')
        return xG_score

    # --------------------------------------------------------------
    # function to return list of cumulative sum for list of numerics
    # --------------------------------------------------------------
    def nums_cumulative_sum(self, num_list):
        return [sum(num_list[:i + 1]) for i in range(len(num_list))]

    # ----------------------------------
    # a static (non-interactive) shotmap
    # ----------------------------------
    def static_shotmap(self, pitch_type='mplsoccer', point_size_range=(20, 500),
                       markers={'SavedShot': "^", 'MissedShots': 'o', 'BlockedShot': "v", 'Goal': '*',
                                'OwnGoal': 'X', 'ShotOnPost': "h"}, pitch_col='#85cb90', line_col='white',
                       alpha=0.5, color1='red', color2='blue',
                       xg_text=True, xg_text_x=None, xg_text_y=None,
                       result_text=True, result_text_x=None, result_text_y=None,
                       name_text=True, name_text_x=None, name_text_y=None,
                       home_image=None, away_image=None, logo_x=None, logo_y=None):

        # create pitch!
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
        # scatter shots
        sns.scatterplot(data=self.data, x=self.x_col, y=self.y_col, hue=self.team_column,
                        style=self.result_col, size=self.xg_col, sizes=point_size_range,
                        markers=markers, alpha=alpha, legend=False,
                        palette={self.home_team: color1, self.away_team: color2})

        # some text informations
        xmax = max(self.x_range_pitch)
        ymax = max(self.y_range_pitch)
        if xg_text:
            # determine positions if not specified
            if pitch_type == 'mplsoccer':
                if xg_text_x is None:
                    xg_text_x = (1 / 3, 2 / 3)
                if xg_text_y is None:
                    xg_text_y = (0.8, 0.8)
            elif pitch_type == 'myPitch':
                if xg_text_x is None:
                    xg_text_x = (1 / 3, 2 / 3)
                if xg_text_y is None:
                    xg_text_y = (0.2, 0.2)
            elif (xg_text_x is None) and (xg_text_y is None):
                raise ValueError(f'If you want the expected goals to be placed as a text, you have to '
                                 f'either select a pitch_type, currently {pitch_type}, or specifiy'
                                 f'the intended positioning via "xg_text_x" and "xg_text_y", currently '
                                 f'{xg_text_x}, {xg_text_y}!')

            plt.text(x=xg_text_x[0] * xmax, y=xg_text_y[0] * ymax,
                     s=str(round(self.xG_score(team='home'), 2)),
                     fontsize=40, weight='bold', c=color1, alpha=0.25, ha='center', va='center')
            plt.text(x=xg_text_x[1] * xmax, y=xg_text_y[1] * ymax,
                     s=str(round(self.xG_score(team='away'), 2)),
                     fontsize=40, weight='bold', c=color2, alpha=0.25, ha='center', va='center')
        if result_text:
            # determine positions if not specified
            if pitch_type == 'mplsoccer':
                if result_text_x is None:
                    result_text_x = (1 / 3, 2 / 3)
                if result_text_y is None:
                    result_text_y = (0.5, 0.5)
            elif pitch_type == 'myPitch':
                if result_text_x is None:
                    result_text_x = (1 / 3, 2 / 3)
                if result_text_y is None:
                    result_text_y = (0.5, 0.5)
            elif (result_text_x is None) and (result_text_y is None):
                raise ValueError(f'If you want the expected goals to be placed as a text, you have to '
                                 f'either select a pitch_type, currently {pitch_type}, or specifiy'
                                 f'the intended positioning via "result_text_x" and "result_text_y", currently '
                                 f'{result_text_x}, {result_text_y}!')

            # determine goal count
            ng1 = self.count_goals(team='home')
            ng2 = self.count_goals(team='away')

            plt.text(x=result_text_x[0] * xmax, y=result_text_y[0] * ymax, s=str(ng1), fontsize=50, weight='bold',
                     c=color1, alpha=0.35,
                     ha='center', va='center')
            plt.text(x=result_text_x[1] * xmax, y=result_text_y[1] * ymax, s=str(ng2), fontsize=50, weight='bold',
                     c=color2, alpha=0.35,
                     ha='center', va='center')

            if name_text:
                # determine positions if not specified
                if pitch_type == 'mplsoccer':
                    if name_text_x is None:
                        name_text_x = (1 / 3, 2 / 3)
                    if name_text_y is None:
                        name_text_y = (0.3, 0.3)
                elif pitch_type == 'myPitch':
                    if name_text_x is None:
                        name_text_x = (1 / 3, 2 / 3)
                    if name_text_y is None:
                        name_text_y = (0.7, 0.65)
                elif (name_text_x is None) and (name_text_y is None):
                    raise ValueError(f"If you want the teams' names to be placed as a text, you have to "
                                     f'either select a pitch_type, currently {pitch_type}, or specify'
                                     f'the intended positioning via "name_text_x" and "name_text_y", currently '
                                     f'{name_text_x}, {name_text_y}!')

                plt.text(x=name_text_x[0] * xmax, y=name_text_y[0] * ymax, s=self.home_team, size='x-large',
                         weight='bold', c=color1, alpha=0.25,
                         ha='center')
                plt.text(x=name_text_x[1] * xmax, y=name_text_y[1] * ymax, s=self.away_team, size='x-large',
                         weight='bold', c=color2, alpha=0.25,
                         ha='center')

            if home_image is not None or away_image is not None:
                # determine positions if not specified
                if pitch_type == 'mplsoccer':
                    if logo_x is None:
                        logo_x = (1 / 3, 2 / 3)
                    if logo_y is None:
                        logo_y = (0.15, 0.15)
                elif pitch_type == 'myPitch':
                    if logo_x is None:
                        logo_x = (1 / 3, 2 / 3)
                    if logo_y is None:
                        logo_y = (0.85, 0.85)
                elif (logo_x is None) and (logo_y is None):
                    raise ValueError(f"If you want the teams' logos displayed, you have to "
                                     f'either select a pitch_type, currently {pitch_type}, or specify'
                                     f'the intended positioning via "home_image_x" and "home_image_y", currently '
                                     f'{logo_x}, {logo_y}!')
                if home_image is not None:
                    Logo_H = mpimg.imread(home_image)
                    imagebox_H = OffsetImage(Logo_H, zoom=0.025)
                    ab_H = AnnotationBbox(imagebox_H, (logo_x[0] * xmax, logo_y[0] * ymax), frameon=False)
                    ax.add_artist(ab_H)
                if away_image is not None:
                    Logo_A = mpimg.imread(away_image)
                    imagebox_A = OffsetImage(Logo_A, zoom=0.025)
                    ab_A = AnnotationBbox(imagebox_A, (logo_x[1] * xmax, logo_y[1] * ymax), frameon=False)
                    ax.add_artist(ab_A)
        return fig

    # ----------------------------------
    # an interactive shotmap with plotly
    # ----------------------------------
    def interactive_shotmap(self, color1='red', color2='blue', pitch_type='mplsoccer', background_col='#16745b',
                            pitch_x0=None, pitch_y0=None, size_multiplicator=5, title=None, title_col='white',
                            xg_text=True, xg_text_x=None, xg_text_y=None, margins=None,
                            result_text=True, result_text_x=None, result_text_y=None,
                            name_text=True, name_text_x=None, name_text_y=None,
                            home_image=None, away_image=None, logo_x=None, logo_y=None,
                            axis_visible=False, pitch_path='', pitch_col='#85cb90', line_col='white',):
        supported_pitch_types = ['mplsoccer', 'myPitch']
        if self.data_source == 'Understat':
            markers = {'SavedShot': "triangle-up", 'MissedShots': 'circle', 'BlockedShot': "triangle-down",
                       'Goal': 'star',
                       'OwnGoal': 'X', 'ShotOnPost': "hexagon"}
        elif self.data_source == 'Statsbomb':
            markers = {'Saved': "triangle-up", 'Saved Off T': "triangle-up", 'Saved To Post': "triangle-up",
                       'Off T': 'circle', 'Wayward': "circle", 'Blocked': "triangle-down", 'Goal': 'star',
                       'OwnGoal': 'X', 'Post': "hexagon"}
        # get symbols / markers in correct order based on fixed dictionary
        symbols = [markers[result] for result in self.data[self.result_col].unique()]

        # create scatter with plotly express
        fig = px.scatter(self.data, x=self.x_col, y=self.y_col, color=self.team_column,
                         symbol=self.result_col, size=self.xg_col,
                         color_discrete_sequence=[color1, color2],
                         custom_data=[self.player_col, self.xg_col, self.result_col],
                         hover_data=[self.xg_col, self.result_col], symbol_sequence=symbols)

        # adapt hover to show player name, xG and result of shot
        fig.update_traces(hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>%{customdata[2]}", hoverinfo=None)

        # create pitch and save a png of pitch
        # also define necessary parameters (origin and size of pitch) if not supplied explicitly
        if pitch_type == 'mplsoccer':
            aa = 'reversed'
            if pitch_x0 is None:
                pitch_x0 = -3.75
            if pitch_y0 is None:
                pitch_y0 = -3.75

            # 0 + 120 + (-2*(0-3.75)) = 127.5 | 0 + 105 + (-2*(0-5.25)) = 115.5
            pitch_sx = self.x_range_pitch[0] + self.x_range_pitch[1] + (-2 * (min(self.x_range_pitch) + pitch_x0))
            # 0 + 65 + (-2*(65 - 68.2)) = 71.4 | 80 + 0 (-2*(0 -3.75)) = 87.5
            pitch_sy = self.y_range_pitch[0] + self.y_range_pitch[1] + (-2 * (min(self.y_range_pitch) + pitch_y0))

            pitch = Pitch(pitch_color=pitch_col, line_color=line_col, stripe=True)
            fig_p, ax = pitch.draw()
        elif pitch_type == 'myPitch':
            aa = True
            if pitch_x0 is None:
                pitch_x0 = -5.25
            if pitch_y0 is None:
                pitch_y0 = 68.2
            pitch_sx = self.x_range_pitch[0] + self.x_range_pitch[1] + (-2 * (min(self.x_range_pitch) + pitch_x0))
            pitch_sy = self.y_range_pitch[0] + self.y_range_pitch[1] + (-2 * (min(self.y_range_pitch) + pitch_y0))
            pitch = myPitch(grasscol=pitch_col, x_range_pitch=self.x_range_pitch, y_range_pitch=self.y_range_pitch)
            fig_p, ax = plt.subplots()
            pitch.plot_pitch(ax=ax)
        else:
            raise ValueError(f'You have to select a valid pitch type out of {supported_pitch_types} '
                             f'so that a pitch can be plotted!')
        pitch_path = pitch_path + 'pitch.png'
        fig_p.savefig(pitch_path, format='png', bbox_inches='tight', pad_inches=0)
        # load pitch as image
        img = Image.open(pitch_path)

        # add the image as background x and y should be subtracting the border of the pitch included in the image so
        # that 0,0 is at the corner of the actual pitch
        # this then requires an adaption of the size of the image by usually twice what was entered at x and y
        fig.add_layout_image(
            dict(source=img, xref='x', yref='y', x=pitch_x0, y=pitch_y0, sizex=pitch_sx, sizey=pitch_sy,
                 sizing='stretch', opacity=1, layer='below'))

        # needed for positioning of added elements
        xmax = max(self.x_range_pitch)
        ymax = max(self.y_range_pitch)

        # determine margins
        if margins is None:
            if pitch_type == 'mplsoccer':
                margins = dict(l=0, r=0, b=10, t=30)
            elif pitch_type == 'myPitch':
                margins = dict(l=10, r=10, b=10, t=30)
            else:
                raise ValueError(f'You need to either select a supported pitch type from {supported_pitch_types} '
                                 f'or specify the intended margins around the plot to "margins"')

        # update layout
        # width and height as multiple of pitch dimensions to maintain aspect ratio
        # margins of paper around actual plot (keep space for header if intended
        fig.update_layout(autosize=True, width=105 * size_multiplicator, height=68 * size_multiplicator,
                          margin=margins,
                          xaxis=dict(visible=axis_visible, autorange=True),
                          yaxis=dict(visible=axis_visible, autorange=aa),
                          title=title, title_font=dict(color=title_col, size=25), title_x=0.5, title_y=0.975,
                          plot_bgcolor=background_col, paper_bgcolor=background_col, showlegend=False)

        # add scatters in the corners = the easiest way to scale plot
        fig.add_trace(go.Scatter(x=[self.x_range_pitch[0], self.x_range_pitch[1]],
                                 y=[self.y_range_pitch[0], self.y_range_pitch[1]], hoverinfo='skip', mode='markers',
                                 marker=dict(size=0, color='red ', opacity=0)))

        # if selected display team logos
        if home_image is not None or away_image is not None:
            # determine positions if not specified
            if pitch_type == 'mplsoccer':
                if logo_x is None:
                    logo_x = (0.4, 0.6)
                if logo_y is None:
                    logo_y = (0.025, 0.025)
            elif pitch_type == 'myPitch':
                if logo_x is None:
                    logo_x = (0.4, 0.6)
                if logo_y is None:
                    logo_y = (0.975, 0.975)
            elif (logo_x is None) and (logo_y is None):
                raise ValueError(f"If you want the teams' logos displayed, you have to "
                                 f'either select a pitch_type out of {supported_pitch_types}, currently {pitch_type},'
                                 f' or specify the intended positioning via "home_image_x" and "home_image_y", '
                                 f'currently {logo_x}, {logo_y}!')

        if home_image is not None:
            himg = Image.open(home_image)
            fig.add_layout_image(
                dict(source=himg, xref='x', yref='y', x=logo_x[0] * xmax, y=logo_y[0] * ymax, sizex=15, sizey=15,
                     xanchor='center', sizing='stretch', opacity=0.9, layer='below'))

        if away_image is not None:
            aimg = Image.open(away_image)
            fig.add_layout_image(
                dict(source=aimg, xref='x', yref='y', x=logo_x[1] * xmax, y=logo_y[1] * ymax, sizex=15, sizey=15,
                     xanchor='center', sizing='stretch', opacity=0.9, layer='below'))

        # if selected display team names
        if name_text:
            # determine positions if not specified
            if pitch_type == 'mplsoccer':
                if name_text_x is None:
                    name_text_x = (1 / 3, 2 / 3)
                if name_text_y is None:
                    name_text_y = (0.3, 0.3)
            elif pitch_type == 'myPitch':
                if name_text_x is None:
                    name_text_x = (1 / 3, 2 / 3)
                if name_text_y is None:
                    name_text_y = (0.72, 0.68)
            elif (name_text_x is None) and (name_text_y is None):
                raise ValueError(f"If you want the teams' names to be placed as a text, you have to "
                                 f'either select a pitch_type from {supported_pitch_types}, currently {pitch_type}, '
                                 f'or specify the intended positioning via "name_text_x" and "name_text_y", currently '
                                 f'{name_text_x}, {name_text_y}!')

            fig.add_annotation(x=name_text_x[0] * xmax, y=name_text_y[0] * ymax, text=self.home_team, showarrow=False,
                               font=dict(family="Arial", size=20, color=color1), opacity=0.75)
            fig.add_annotation(x=name_text_x[1] * xmax, y=name_text_y[1] * ymax, text=self.away_team, showarrow=False,
                               font=dict(family="Arial", size=20, color=color2), opacity=0.75)

        # if selected display actual result
        if result_text:
            # determine positions if not specified
            if pitch_type == 'mplsoccer':
                if result_text_x is None:
                    result_text_x = (0.4, 0.6)
                if result_text_y is None:
                    result_text_y = (0.5, 0.5)
            elif pitch_type == 'myPitch':
                if result_text_x is None:
                    result_text_x = (0.4, 0.6)
                if result_text_y is None:
                    result_text_y = (0.5, 0.5)
            elif (result_text_x is None) and (result_text_y is None):
                raise ValueError(f'If you want the expected goals to be placed as a text, you have to '
                                 f'either select a pitch_type from {supported_pitch_types}, currently {pitch_type}, '
                                 f'or specify the intended positioning via "result_text_x" and "result_text_y", '
                                 f'currently {result_text_x}, {result_text_y}!')

            # determine goal count
            ng1 = self.count_goals(team='home')
            ng2 = self.count_goals(team='away')
            fig.add_annotation(x=result_text_x[0] * xmax, y=result_text_y[0] * ymax, text=ng1, showarrow=False,
                               font=dict(family="Arial", size=100, color=color1), opacity=0.75)
            fig.add_annotation(x=result_text_x[1] * xmax, y=result_text_y[1] * ymax, text=ng2, showarrow=False,
                               font=dict(family="Arial", size=100, color=color2), opacity=0.75)

        if xg_text:
            # determine positions if not specified
            if pitch_type == 'mplsoccer':
                if xg_text_x is None:
                    xg_text_x = (0.4, 0.6)
                if xg_text_y is None:
                    xg_text_y = (0.8, 0.8)
            elif pitch_type == 'myPitch':
                if xg_text_x is None:
                    xg_text_x = (0.4, 0.6)
                if xg_text_y is None:
                    xg_text_y = (0.2, 0.2)
            elif (xg_text_x is None) and (xg_text_y is None):
                raise ValueError(f'If you want the expected goals to be placed as a text, you have to '
                                 f'either select a pitch_type from {supported_pitch_types}, currently {pitch_type}, '
                                 f'or specify the intended positioning via "xg_text_x" and "xg_text_y", currently '
                                 f'{xg_text_x}, {xg_text_y}!')

            xg1 = str(round(self.xG_score(team='home'), 2))
            xg2 = str(round(self.xG_score(team='away'), 2))
            fig.add_annotation(x=xg_text_x[0] * xmax, y=xg_text_y[0] * ymax, text=xg1, showarrow=False,
                               font=dict(family="Arial", size=40, color=color1), opacity=0.5)
            fig.add_annotation(x=xg_text_x[1] * xmax, y=xg_text_y[1] * ymax, text=xg2, showarrow=False,
                               font=dict(family="Arial", size=40, color=color2), opacity=0.5)

        return fig

    # --------------------------------
    # function to create xG flow chart
    # --------------------------------

    def xg_chart(self, color1='red', color2='blue', Title=None, text_col='white', font_type='Rockwell',
                 grid_visible=True, grid_col='#a3a3a3', plot_col='#999999', ball_image_path='images/Football3.png',
                 display_score=True, home_image=None, away_image=None, design=None, ball_size_x=1.75, ball_size_y=0.1):
        if design == 'light':
            text_col = 'black'
            grid_col = '#CDE1ED'
            plot_col = '#f8f8ff'
        if design == 'spotify':  # inspired by https://www.color-hex.com/color-palette/53188
            plot_col = '#212121'
            grid_col = '#121212'
            text_col = '#b3b3b3'
            color1 = '#1db954'
            color2 = 'white'
        if design == 'dark':
            plot_col = '#212121'
            grid_col = '#121212'
            text_col = '#b3b3b3'
        if design == 'fun':
            ball_image_path = 'images/Kick.png'
            ball_size_x = 3
            ball_size_y = 0.175
            plot_col = '#e2eeff'
            text_col = '#0f403f'
            grid_col = '#ffdef2'

        # use the org data for xG with all decimals
        df = self.data
        # min as integer and xG as float
        df[self.minute_col] = df[self.minute_col].astype(int)
        df[self.xg_col] = df[self.xg_col].astype(float)

        # adding a row at start and end for lines to cover the entire playing time!
        extra_row_home = pd.Series({self.xg_col: 0, self.minute_col: 0, self.team_column: self.home_team})
        extra_row_away = pd.Series({self.xg_col: 0, self.minute_col: 0, self.team_column: self.away_team})
        minute = max(df[self.minute_col]) + 1
        extra_row_home2 = pd.Series({self.xg_col: 0, self.minute_col: minute, self.team_column: self.home_team,
                                     self.result_col: 'Final whistle', self.player_col: ''})
        extra_row_away2 = pd.Series({self.xg_col: 0, self.minute_col: minute, self.team_column: self.away_team,
                                     self.result_col: 'Final whistle', self.player_col: ''})
        df = pd.concat([df, extra_row_home.to_frame().T, extra_row_away.to_frame().T, extra_row_home2.to_frame().T,
                        extra_row_away2.to_frame().T], ignore_index=True)

        # re-attribute own goals to the scoring team
        if "OwnGoal" in df[self.result_col].values:
            OGs = np.where(df[self.result_col] == "OwnGoal")[0]
            for og in OGs:
                df[self.team_column][og] = self.home_team if df[self.team_column][
                                                                 og] == self.away_team else self.away_team

        # sort by team and minute within team
        order_by_custom = pd.CategoricalDtype([self.home_team, self.away_team], ordered=True)
        df[self.team_column] = df[self.team_column].astype(order_by_custom)
        df.sort_values(by=[self.team_column, self.minute_col], inplace=True)

        # specify color vector
        colors = [color1, color2]

        # cumulative sum of each team's xG for y-axis
        home_cumulative_xG = self.nums_cumulative_sum(df[self.xg_col][df[self.team_column] == self.home_team].to_list())
        away_cumulative_xG = self.nums_cumulative_sum(df[self.xg_col][df[self.team_column] == self.away_team].to_list())
        # adding the cumulative xG including 0
        df['xG_cum'] = home_cumulative_xG + away_cumulative_xG

        # plot lines
        fig = px.line(df, x=self.minute_col, y='xG_cum', color=self.team_column, line_shape='hv',
                      color_discrete_sequence=colors, labels=dict(minute='Minute', xG_cum="xGoals", team="Team"),
                      hover_name="player", hover_data={self.xg_col: ':.2f', self.minute_col: False,
                                                       self.result_col: True, self.team_column: False, 'xG_cum': False})

        # update layout
        fig.update_layout(plot_bgcolor=plot_col, paper_bgcolor=plot_col, showlegend=False,
                          title={'text': Title, 'y': 0.95, 'x': 0.5, 'yanchor': 'top', 'xanchor': 'center',
                                 'font_size': 20},
                          font={'family': font_type, 'color': text_col},
                          hoverlabel={'font_size': 16},
                          xaxis=dict(tickmode='linear', tick0=0, dtick=15, range=[0, max(df[self.minute_col])],
                                     showgrid=grid_visible, showline=False, zeroline=False, tickcolor=grid_col,
                                     gridcolor=grid_col),
                          yaxis=dict(tickmode='linear', tick0=0, dtick=0.5, range=[0, round(max(df['xG_cum'])) + 0.5],
                                     showgrid=grid_visible, showline=False, zeroline=False, tickcolor=grid_col,
                                     gridcolor=grid_col)
                          )

        # adding balls as images where a goal was scored
        ball = Image.open(ball_image_path)

        for i in range(len(df[self.minute_col])):
            if df[self.result_col][i] == "Goal" or df[self.result_col][i] == "OwnGoal":
                fig.add_layout_image(
                    dict(source=ball, xref='x', yref='y', x=df[self.minute_col][i], y=df['xG_cum'][i],
                         sizex=ball_size_x, sizey=ball_size_y, xanchor='center', yanchor='middle', sizing='stretch',
                         opacity=1,
                         layer='above'))

        # display final score if selcted
        if display_score:
            # determine goal count
            ng1 = self.count_goals(team='home')
            ng2 = self.count_goals(team='away')

            fig.add_annotation(x=7.5, y=round(max(df.xG_cum)) - 0.25, text=ng1, showarrow=False,
                               font=dict(family='Rockwell', size=40, color=color1), opacity=1)
            fig.add_annotation(x=22.5, y=round(max(df.xG_cum)) - 0.25, text=ng2, showarrow=False,
                               font=dict(family='Rockwell', size=40, color=color2), opacity=1)

        # display team logos if supplied
        if home_image is not None and away_image is not None:
            himg = Image.open(home_image)
            aimg = Image.open(away_image)
            fig.add_layout_image(dict(source=himg, xref='x', yref='y', x=7.5, y=round(max(df.xG_cum)) + 0.25,
                                      sizex=15, sizey=0.5, xanchor='center', yanchor='middle', sizing='stretch',
                                      opacity=0.9, layer='below'))
            fig.add_layout_image(dict(source=aimg, xref='x', yref='y', x=22.5, y=round(max(df.xG_cum)) + 0.25,
                                      sizex=15, sizey=0.5, xanchor='center', yanchor='middle', sizing='stretch',
                                      opacity=0.9, layer='below'))

        return fig
