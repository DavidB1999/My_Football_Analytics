import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from Pitch.My_Pitch import \
    myPitch  # might need adaptation of path depending on whether it is used in pycharm or jupyter notebook
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import numpy as np


# ------------------------------------------------------------------------
# pass data as its own class with functions to rescale and create shot map
# ------------------------------------------------------------------------

class pass_data():

    def __init__(self, data, data_source, x_range_data=None, y_range_data=None, team_column='team',
                 location_column=None, end_location_key=None, pass_column=None,
                 scale_to_pitch='mplsoccer', x_range_pitch=None, y_range_pitch=None,
                 mirror_away=['x', 'y'], columns_to_keep=None):
        self.org_data = data
        self.data_source = data_source
        self.x_range_data = x_range_data
        self.y_range_data = y_range_data
        self.team_column = team_column
        self.scale_to_pitch = scale_to_pitch
        self.x_range_pitch = x_range_pitch
        self.y_range_pitch = y_range_pitch
        self.location_column = location_column
        self.end_location_key = end_location_key
        self.pass_column = pass_column
        self.mirror_away = mirror_away
        self.columns_to_keep = columns_to_keep

        supported_data_source = ['Statsbomb']
        # usually in Statsbomb both teams play both halves left to right (0 to 120)
        if self.data_source == 'Statsbomb':
            if self.location_column is None:
                self.location_column = 'location'
            if self.end_location_key is None:
                self.end_location_key = 'end_location'
            if self.pass_column is None:
                self.pass_column = 'pass'
            if self.x_range_data is None:
                self.x_range_data = (0, 120)
            if self.y_range_data is None:
                self.y_range_data = (80, 0)
        elif self.x_range_data is None or self.y_range_data is None:
            raise ValueError(f'You have not selected a data source which which would indicate an original scale.'
                             f'Neither did you supply custom ranges via "x_range_data" and "y_range_data"'
                             f'Either supply one of {supported_data_source} to "data_source" or '
                             f'Supply tuples of data ranges to "x_range_data" and "x_range_data".')

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

        # on initializing the data is rescaled, but I can always initialize again based on org_data!
        self.data = self.rescale_pass_data()

    def rescale_pass_data(self):

        # deep copy to avoid changes on original data if not intended
        data = self.org_data.copy(deep=True)

        if self.x_range_data is None or self.y_range_data is None or self.x_range_pitch is None or self.y_range_pitch is None:
            raise ValueError(f'Oops! Something went wrong. The coordinates for rescaling are missing.')

        # start and end location split into separate lists for both x and y
        # loop over pass location and access both x and y
        x1_org = []
        x2_org = []
        y1_org = []
        y2_org = []

        for p in data[self.location_column]:
            x1_org.append(float(p[0]))
            y1_org.append(float(p[1]))
        for p in data[self.pass_column]:  # access the dictionary stored at column 'pass' and get end_location by key
            el = p[self.end_location_key]
            x2_org.append(float(el[0]))
            y2_org.append(float(el[1]))
