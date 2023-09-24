import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from Pitch.My_Pitch import myPitch  # might need adaptation of path depending on whether it is used in pycharm or jupyter notebook
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image



# ------------------------------------------------------------------------
# shot data as its own class with functions to rescale and create shot map
# ------------------------------------------------------------------------

class shot_data:
    def __init__(self, data, data_source=None, x_range_data=None, y_range_data=None, team_column='team',
                 x_col='x', y_col='y', xg_col='xG', minute_col='minute', result_col='result',
                 scale_to_pitch='mplsoccer', x_range_pitch=None, y_range_pitch=None, player_col = 'player',
                 mirror_away=['x', 'y']):
        self.org_data = data
        self.data_source = data_source
        self.x_range_data = x_range_data
        self.y_range_data = y_range_data
        self.team_column = team_column
        self.x_col = x_col
        self.y_col = y_col
        self.xg_col = xg_col
        self.minute_col = minute_col
        self.result_col = result_col
        self.player_col = player_col
        self.home_team = data[self.team_column].unique()[0]
        self.away_team = data[self.team_column].unique()[1]
        self.filter1 = data[self.team_column] == self.home_team
        self.filter2 = data[self.team_column] == self.away_team
        self.scale_to_pitch = scale_to_pitch
        self.x_range_pitch = x_range_pitch
        self.y_range_pitch = y_range_pitch
        self.mirror_away = mirror_away
        # on initializing the data is rescaled but i can alaway initialize again based on org_data!
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
        # supported pitch types have a default data range to which data can be rescaled
        supported_pitch_types = ['mplsoccer', 'myPitch']
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
                             f'Either supply one of {supported_pitch_types} to "scale_to_pitch" or '
                             f'Supply tuples of data ranges to "x_range_pitch" and "y_range_pitch".')
        supported_data_source = ['Understat']
        if self.data_source == 'Understat':
            if self.x_range_data is None:
                self.x_range_data = (0, 1)
            else:
                self.x_range_data = self.x_range_data
            if self.y_range_data is None:
                self.y_range_data = (0, 1)
            else:
                self.y_range_data = self.y_range_data
        elif self.x_range_data is None or self.y_range_data is None:
            raise ValueError(f'You have not selected a data source which which would indicate an original scale.'
                             f'Neither did you supply custom ranges via "x_range_data" and "y_range_data"'
                             f'Either supply one of {supported_data_source} to "data_source" or '
                             f'Supply tuples of data ranges to "x_range_data" and "x_range_data".')

        # this works without data_source being supplied but this nevertheless
        # requires a format similar to understat shot data
        if self.data_source == 'Understat' or (self.data_source is None):

            # convert to float / int
            data[self.x_col] = data[self.x_col].astype(float)
            data[self.y_col] = data[self.y_col].astype(float)
            data[self.xg_col] = round(data[self.xg_col].astype(float), 2)
            data[self.minute_col] = data[self.minute_col].astype(int)

            # dimensions dictionary for convenience
            dimensions = {'y': {
                'data': self.y_range_data,
                'pitch': self.y_range_pitch},
                'x': {
                    'data': self.x_range_data,
                    'pitch': self.x_range_pitch}
            }
            # for both the x and y coordinates
            for dim in dimensions.keys():
                data0 = dimensions[dim]['data'][0]
                data1 = dimensions[dim]['data'][1]
                delta_data = data1 - data0
                pitch0 = dimensions[dim]['pitch'][0]
                pitch1 = dimensions[dim]['pitch'][1]
                delta_pitch = pitch1 - pitch0
                # print(delta_data, delta_pitch)
                scaling_factor = delta_pitch / delta_data
                # print(scaling_factor)
                # print(data0)
                # print(data1)

                # if the data we want to rescale, is oriented in the usual direction (left to right, bottom to top)
                # we calculate like this
                if delta_data > 0:
                    # rescale home team coordinates
                    data.loc[self.filter1, dim] = data.loc[self.filter1, dim].apply(
                        lambda x: pitch0 + x * scaling_factor)

                    # rescale away team and if necessary mirror
                    if dim in self.mirror_away:
                        data.loc[self.filter2, dim] = data.loc[self.filter2, dim].apply(
                            lambda x: pitch1 - x * scaling_factor)
                    else:
                        data.loc[self.filter2, dim] = data.loc[self.filter2, dim].apply(
                            lambda x: pitch0 + x * scaling_factor)

                # if the data we want to rescale is mirrored in dim
                # we calculate like this
                elif delta_data < 0:
                    # rescale home team coordinates
                    data.loc[self.filter1, dim] = data.loc[self.filter1, dim].apply(
                        lambda x: pitch1 + x * scaling_factor)

                    # rescale away team and if necessary mirror
                    if dim in self.mirror_away:
                        data.loc[self.filter2, dim] = data.loc[self.filter2, dim].apply(
                            lambda x: pitch0 - x * scaling_factor)
                    else:
                        data.loc[self.filter2, dim] = data.loc[self.filter2, dim].apply(
                            lambda x: pitch1 + x * scaling_factor)
        else:  # currently "Understat" is the only option included (could add others like Statsbomb)
            raise ValueError(
                f'{self.data_source} not supported. At this point, "Understat" is the only supported data format.')

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

    # ----------------------------------
    # a static (non-interactive) shotmap
    # ----------------------------------
    def static_shotmap(self, pitch_type='mplsoccer', point_size_range=(20, 500),
                       markers={'SavedShot': "^", 'MissedShots': 'o', 'BlockedShot': "v", 'Goal': '*',
                                'OwnGoal': 'X', 'ShotOnPost': "h"},
                       alpha=0.5, color1='red', color2='blue',
                       xg_text=True, xg_text_x=None, xg_text_y=None,
                       result_text=True, result_text_x=None, result_text_y=None,
                       name_text=True, name_text_x=None, name_text_y=None,
                       home_image=None, away_image=None, logo_x=None, logo_y=None):

        # create pitch!
        if pitch_type == 'mplsoccer':
            pitch = Pitch(pitch_color='grass', line_color='white', stripe=True)
            fig, ax = pitch.draw()
        elif pitch_type == 'myPitch':
            pitch = myPitch()
            fig, ax = plt.subplots()
            pitch.plot_pitch(ax=ax)

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
                     s=str(self.xG_score(team='home')),
                     fontsize=40, weight='bold', c=color1, alpha=0.25, ha='center', va='center')
            plt.text(x=xg_text_x[1] * xmax, y=xg_text_y[1] * ymax,
                     s=str(self.xG_score(team='away')),
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

    # ----------------------------------
    # an interactive shotmap with plotly
    # ----------------------------------
    def interactive_shotmap(self, color1='red', color2='blue', pitch_type='mplsoccer',
                            pitch_x0=None, pitch_y0=None):

        markers = {'SavedShot': "triangle-up", 'MissedShots': 'circle', 'BlockedShot': "triangle-down", 'Goal': 'star',
                   'OwnGoal': 'X', 'ShotOnPost': "hexagon"}

        # get symbols / markers in correct order based on fixed dictionary
        symbols = []
        for result in self.data[self.result_col].unique():
            symbols.append(markers[result])

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
            if pitch_x0 is None:
                pitch_x0 = -3.75
            if pitch_y0 is None:
                pitch_y0 = -3.75

            # 0 + 120 + (-2*(0-3.75)) = 127.5 | 0 + 105 + (-2*(-5.25)) = 115.5
            pitch_sx = self.x_range[0] + self.x_range[1] + (-2*(self.x_range[0]-pitch_x0))
            # 0 + 65 + (-2*(65 - 68.2)) = 71.4 | 80 + 0 (-2*(0 -3.75)) = 87.5
            pitch_sy = self.y_range[0] + self.x_range[1] + (-2*(self.y_range[0]-pitch_x0))

            pitch = Pitch(pitch_color='grass', line_color='white', stripe=True)
            fig, ax = pitch.draw()
        elif pitch_type == 'myPitch':
            if pitch_x0 is None:
                pitch_x0 = -5.25
            if pitch_y0 is None:
                pitch_y0 = 68.2

            pitch_sx = self.x_range[0] + self.x_range[1] + (-2*(self.x_range[0]-pitch_x0))
            pitch_sy = self.y_range[0] + self.x_range[1] + (-2*(self.y_range[0]-pitch_x0))
            pitch = myPitch()
            fig, ax = plt.subplots()
            pitch.plot_pitch(ax=ax)
        fig.savefig('pitch.png', format='png', bbox_inches='tight', pad_inches=0)
        # load pitch as image
        img = Image.open('pitch.png')

        # add the image as background x and y should be subtracting the border of the pitch included in the image so
        # that 0,0 is at the corner of the actual pitch
        # this then requires an adaption of the size of the image by usually twice what was entered at x and y
        fig.add_layout_image(
            dict(source=img,
                 xref='x',
                 yref='y',
                 x=pitch_x0,
                 y=-pitch_y0,
                 sizex=pitch_sx,
                 sizey=pitch_sy,
                 sizing='stretch',
                 opacity=1,
                 layer='below')
        )



