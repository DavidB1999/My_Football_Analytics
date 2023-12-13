import numpy as np


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


def get_radar_coord():
    """
    Function to get the coordinates of the radar area vertices (for the polygons)
    """
