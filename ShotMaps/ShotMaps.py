import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from Pitch.My_Pitch import myPitch # might need adaptation of path depending on whether it is used in pycharm or jupyter notebook
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox



# ------------------------------------------------------------------------
# shot data as its own class with functions to rescale and create shot map
# ------------------------------------------------------------------------

class shot_data:
    def __init__(self, data, data_source=None, x_range_data=None, y_range_data=None, team_column='team',
                 x_col='x', y_col='y', xg_col='xG', minute_col='minute', result_col='result'):
        self.data = data
        self.data_source = data_source
        self.x_range_data = x_range_data
        self.y_range_data = y_range_data
        self.team_column = team_column
        self.x_col = x_col
        self.y_col = y_col
        self.xg_col = xg_col
        self.minute_col = minute_col
        self.result_col = result_col

    def __str__(self):
        return (
            f"shot_data object of {self.data_source} of shape {self.data.shape}."
        )

    # --------------------------------------------------
    # function to rescale data to a data range of choice
    # --------------------------------------------------
    def rescale_shot_data(self, scale_to_pitch='mplsoccer', x_range_pitch=None, y_range_pitch=None,
                          mirror_away=['x', 'y']):

        # deep copy to avoid changes on original data if not intended
        data = self.data.copy(deep=True)
        # supported pitch types have a default data range to which data can be rescaled
        supported_pitch_types = ['mplsoccer', 'myPitch']
        if scale_to_pitch == 'mplsoccer':
            if (x_range_pitch is None):
                x_range_pitch = (0, 120)
            if (y_range_pitch is None):
                y_range_pitch = (80, 0)
        elif scale_to_pitch == 'myPitch':
            if (x_range_pitch is None):
                x_range_pitch = (0, 105)
            if (y_range_pitch is None):
                y_range_pitch = (0, 65)
        elif (x_range_pitch is None) or (y_range_pitch is None):
            raise ValueError(f'You have not selected a pitch type to which the data is supposed to be scaled.'
                             f'Neither did you supply custom ranges via "x_range_pitch" and "y_range_pitch"'
                             f'Either supply one of {supported_pitch_types} to "scale_to_pitch" or '
                             f'Supply tuples of data ranges to "x_range_pitch" and "y_range_pitch".')

        if self.data_source == 'Understat':
            if (self.x_range_data is None):
                x_range_data = (0, 1)
            else:
                x_range_data = self.x_range_data
            if (self.y_range_data is None):
                y_range_data = (0, 1)
            else:
                y_range_data = self.y_range_data

        # this works without data_source being supplied but this nevertheless
        # requires a format similar to understat shot data
        if self.data_source == 'Understat' or (self.data_source is None):

            # get team identities
            home_team = data[self.team_column].unique()[0]
            away_team = data[self.team_column].unique()[1]
            # to filter them
            filter1 = data[self.team_column] == home_team
            filter2 = data[self.team_column] == away_team

            # convert to float / int
            data[self.x_col] = data[self.x_col].astype(float)
            data[self.y_col] = data[self.y_col].astype(float)
            data[self.xg_col] = round(data[self.xg_col].astype(float), 2)
            data[self.minute_col] = data[self.minute_col].astype(int)

            # dimensions dictionary for convenience
            dimensions = {'y': {
                'data': y_range_data,
                'pitch': y_range_pitch},
                'x': {
                    'data': x_range_data,
                    'pitch': x_range_pitch}
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
                    data.loc[filter1, dim] = data.loc[filter1, dim].apply(lambda x: pitch0 + x * scaling_factor)

                    # rescale away team and if necessary mirror
                    if dim in mirror_away:
                        data.loc[filter2, dim] = data.loc[filter2, dim].apply(lambda x: pitch1 - x * scaling_factor)
                    else:
                        data.loc[filter2, dim] = data.loc[filter2, dim].apply(lambda x: pitch0 + x * scaling_factor)

                # if the data we want to rescale is mirrored in dim
                # we calculate like this
                elif delta_data < 0:
                    # rescale home team coordinates
                    data.loc[filter1, dim] = data.loc[filter1, dim].apply(lambda x: pitch1 + x * scaling_factor)

                    # rescale away team and if necessary mirror
                    if dim in mirror_away:
                        data.loc[filter2, dim] = data.loc[filter2, dim].apply(lambda x: pitch0 - x * scaling_factor)
                    else:
                        data.loc[filter2, dim] = data.loc[filter2, dim].apply(lambda x: pitch1 + x * scaling_factor)
        else:  # currently "Understat" is the only option included (could add others like Statsbomb)
            raise ValueError(
                f'{self.data_source} not supported. At this point, "Understat" is the only supported data format.')

        return data

    # ----------------------------------
    # a static (non-interactive) shotmap
    # ----------------------------------
    def static_shotmap(self, pitch_type='mplsoccer', point_size_range=(20,500),
                       markers={'SavedShot': "^", 'MissedShots': 'o', 'BlockedShot': "v", 'Goal': '*',
                                'OwnGoal': 'X', 'ShotOnPost': "h"},
                       alpha=0.5, color1='red', color2='blue',
                       x_range_pitch=None, y_range_pitch=None, sc_mirror_away=['x', 'y'],
                       xg_text=True, result_text=True, name_text=True,
                       home_image=None, away_image=None):

        # scale using function
        scaled_shot_data = self.rescale_shot_data(scale_to_pitch=pitch_type, x_range_pitch=x_range_pitch,
                                                  y_range_pitch=y_range_pitch, mirror_away=sc_mirror_away)

        # get team identities
        home_team = scaled_shot_data[self.team_column].unique()[0]
        away_team = scaled_shot_data[self.team_column].unique()[1]
        # to filter them
        filter1 = scaled_shot_data[self.team_column] == home_team
        filter2 = scaled_shot_data[self.team_column] == away_team

        # create pitch!
        if pitch_type == 'mplsoccer':
            pitch = Pitch(pitch_color='grass', line_color='white', stripe=True)
            fig, ax = pitch.draw()
        elif pitch_type == 'myPitch':
            pitch = myPitch()
            fig, ax = plt.subplots()
            pitch.plot_pitch(ax=ax)

        # scatter shots
        sns.scatterplot(data=scaled_shot_data, x=self.x_col, y=self.y_col, hue=self.team_column,
                        style=self.result_col, size=self.xg_col, sizes=point_size_range,
                        markers=markers, alpha=alpha, legend=False,
                        palette={home_team: color1, away_team: color2})

        # some text informations
        if xg_text:
            plt.text(x=40, y=60, s=str(sum(scaled_shot_data.loc[filter1, self.xg_col])),
                     fontsize=40, weight='bold', c=color1, alpha=0.25, ha='center', va='center')
            plt.text(x=80, y=60, s=str(sum(scaled_shot_data.loc[filter2, self.xg_col])),
                     fontsize=40, weight='bold', c=color2, alpha=0.25, ha='center', va='center')
        if result_text:
            ng1 = len(scaled_shot_data[self.result_col][scaled_shot_data[self.result_col] == 'Goal'][scaled_shot_data[self.team_column] == home_team])
            + len(scaled_shot_data[self.result_col][scaled_shot_data[self.result_col] == 'OwnGoal'][scaled_shot_data[self.team_column] == away_team])
            ng2 = len(scaled_shot_data[self.result_col][scaled_shot_data[self.result_col] == 'Goal'][
                          scaled_shot_data[self.team_column] == away_team])
            + len(scaled_shot_data[self.result_col][scaled_shot_data[self.result_col] == 'OwnGoal'][
                      scaled_shot_data[self.team_column] == home_team])

            plt.text(x=40, y=40, s=str(ng1), fontsize=50, weight='bold', c=color1, alpha=0.35,
                     ha='center', va='center')
            plt.text(x=80, y=40, s=str(ng2), fontsize=50, weight='bold', c=color2, alpha=0.35,
                     ha='center', va='center')

            if name_text:
                plt.text(x=40, y=22, s=home_team, size='x-large', weight='bold', c=color1, alpha=0.25,
                         ha='center')
                plt.text(x=80, y=22, s=away_team, size='x-large', weight='bold', c=color2, alpha=0.25,
                         ha='center')

            if home_image is not None:
                Logo_H = mpimg.imread(home_image)
                imagebox_H = OffsetImage(Logo_H, zoom=0.025)
                ab_H = AnnotationBbox(imagebox_H, (40, 10), frameon=False)
                ax.add_artist(ab_H)
            if home_image is not None:
                Logo_A = mpimg.imread(away_image)
                imagebox_A = OffsetImage(Logo_A, zoom=0.025)
                ab_A = AnnotationBbox(imagebox_A, (80, 10), frameon=False)
                ax.add_artist(ab_A)







