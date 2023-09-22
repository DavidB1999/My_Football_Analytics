import pandas as pd


# ------------------------------------------------------------
# function to rescale and structure data for shot map function
# ------------------------------------------------------------
def rescale_shot_data(shot_data, data_source=None, x_range_data=None, y_range_data=None, team_column='team',
                      x_col='x', y_col='y', xg_col='xG', minute_col='minute', scale_to_pitch='mplsoccer',
                      x_range_pitch=None, y_range_pitch=None, mirror_away=['x', 'y']):
    # deep copy to avoid changes on original data if not intended
    data = shot_data.copy(deep=True)
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

    if data_source == 'Understat':
        if (x_range_data is None):
            x_range_data = (0, 1)
        if (y_range_data is None):
            y_range_data = (0, 1)

    # this works without data_source being supplied but this nevertheless
    # requires a format similar to understat shot data
    if data_source == 'Understat' or (data_source is None):

        # get team identities
        home_team = data[team_column].unique()[0]
        away_team = data[team_column].unique()[1]
        # to filter them
        filter1 = data[team_column] == home_team
        filter2 = data[team_column] == away_team

        # convert to float / int
        data[x_col] = data[x_col].astype(float)
        data[y_col] = data[y_col].astype(float)
        data[xg_col] = round(data[xg_col].astype(float), 2)
        data[minute_col] = data[minute_col].astype(int)

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
            #print(delta_data, delta_pitch)
            scaling_factor = delta_pitch / delta_data
            #print(scaling_factor)
            #print(data0)
            #print(data1)

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
        raise ValueError(f'{data_source} not supported. At this point, "Understat" is the only supported data format.')

    return data
