import numpy as np
import pandas as pd


def get_label_coordinates(n):
    """
    Function to get coordinates for labels for each polar in radar chart
    """

    # numpy takes angle in radians (2pi = 360 degrees)
    # alpha the angle per param in radians
    alpha = 2 * np.pi / n
    alphas = alpha * np.arange(n)

    ## x-coordinate value
    coord_x = np.cos(alphas)

    ## y-coordinate value
    coord_y = np.sin(alphas)

    return np.c_[coord_x, coord_y, alphas]


def get_radar_coord(values, radius, ranges, rot):
    """
    Function to get the coordinates of the radar area vertices (for the polygons)
    """
    # need to add 1 to the distance from the center due to axis starting at 1
    xy = [[(1 + radius * ((v - rg[0]) / (rg[1] - rg[0]))) * np.cos(rt),
           (1 + radius * ((v - rg[0]) / (rg[1] - rg[0]))) * np.sin(rt)] for v, rg, rt in zip(values, ranges, rot)]

    return xy


def param_select(df, params, var_col='Variables'):
    df_new = df[df[var_col].isin(params)]  # select columns of interest

    # merge with column of parameters in correct order
    # to ensure data is sorted by the order of parameters selected
    dummy = pd.Series(params, name=var_col).to_frame()
    df_new = pd.merge(dummy, df_new, on=var_col, how='left')

    return df_new
