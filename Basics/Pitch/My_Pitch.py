# My own function and class to create a football pitch

# -------------------------
# import necessary packages
# -------------------------
import sys

sys.path.append('C:\\Users\\DavidB\\PycharmProjects\\My_Football_Analytics')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib


# ------------------------------------------------------------------------------------
# pitch dimensions based on https://de.wikipedia.org/wiki/Fu%C3%9Fballregeln in meters
# ------------------------------------------------------------------------------------

# --------------
# Pitch as class
# --------------
class myPitch:
    def __init__(self, l=105, w=65, grasscol='#6aa84f', linecol='white', goal_width=5):
        self.l = l
        self.w = w
        self.grasscol = grasscol
        self.linecol = linecol
        self.goal_width = goal_width

    def __str__(self):
        return (
            f"myPitch object with axes y (i.e. width) = {self.w} / x (i.e. length) = {self.l} "
        )

    # --------------------------------------------------
    # Plots a football pitch on a given matplotlib.axes.
    # --------------------------------------------------
    def plot_pitch(self, ax: matplotlib.axes = None, **kwargs):
        # kwargs which are used to configure the plot with default values 1.5 and 0.
        # all the other kwargs will be just passed to all the plot functions.
        linewidth = kwargs.pop("linewidth", 1.5)
        zorder = kwargs.pop("zorder", 0)

        # check whether an axes to plot is given or if a new axes element has to be created
        ax = ax or plt.subplots()[1]

        ax.patch.set_facecolor(self.grasscol)
        # Pitch Outline & Centre Line
        ax.plot([0, 0], [0, self.w], color=self.linecol, linewidth=linewidth)
        ax.plot([0, self.l], [self.w, self.w], color=self.linecol, linewidth=linewidth)
        ax.plot([self.l, self.l], [self.w, 0], color=self.linecol, linewidth=linewidth)
        ax.plot([self.l, 0], [0, 0], color=self.linecol, linewidth=linewidth)
        ax.plot([self.l / 2, self.l / 2], [0, self.w], color=self.linecol, linewidth=linewidth)

        # Penalty areas - 7.32*0.5 + 5.5 + 11 = 20.16
        # Left
        ax.plot([16.5, 16.5], [self.w / 2 + 20.16, self.w / 2 - 20.16], color=self.linecol, linewidth=linewidth)
        ax.plot([0, 16.5], [self.w / 2 + 20.16, self.w / 2 + 20.16], color=self.linecol, linewidth=linewidth)
        ax.plot([16.5, 0], [self.w / 2 - 20.16, self.w / 2 - 20.16], color=self.linecol, linewidth=linewidth)
        # Right
        ax.plot([self.l, self.l - 16.5], [self.w / 2 + 20.16, self.w / 2 + 20.16], color=self.linecol,
                linewidth=linewidth)
        ax.plot([self.l - 16.5, self.l - 16.5], [self.w / 2 + 20.16, self.w / 2 - 20.16], color=self.linecol,
                linewidth=linewidth)
        ax.plot([self.l - 16.5, self.l], [self.w / 2 - 20.16, self.w / 2 - 20.16], color=self.linecol,
                linewidth=linewidth)

        # 6 yard boxes - 7.32*0.5 + 5.5 = 9.16
        # Left
        ax.plot([0, 5.5], [self.w / 2 + 9.16, self.w / 2 + 9.16], color=self.linecol, linewidth=linewidth)
        ax.plot([5.5, 5.5], [self.w / 2 + 9.16, self.w / 2 - 9.16], color=self.linecol, linewidth=linewidth)
        ax.plot([5.5, 0.5], [self.w / 2 - 9.16, self.w / 2 - 9.16], color=self.linecol, linewidth=linewidth)
        # Right
        ax.plot([self.l, self.l - 5.5], [self.w / 2 + 9.16, self.w / 2 + 9.16], color=self.linecol, linewidth=linewidth)
        ax.plot([self.l - 5.5, self.l - 5.5], [self.w / 2 + 9.16, self.w / 2 - 9.16], color=self.linecol,
                linewidth=linewidth)
        ax.plot([self.l - 5.5, self.l], [self.w / 2 - 9.16, self.w / 2 - 9.16], color=self.linecol, linewidth=linewidth)

        # Goals - 7.32*0.5
        ax.plot([self.l, self.l], [self.w / 2 - 3.66, self.w / 2 + 3.66], color=self.linecol, linewidth=self.goal_width)
        ax.plot([0, 0], [self.w / 2 - 3.66, self.w / 2 + 3.66], color=self.linecol, linewidth=self.goal_width)

        # Prepare Circles
        centreCircle = plt.Circle((self.l / 2, self.w / 2), 9.15, color=self.linecol, fill=False, linewidth=linewidth)
        centreSpot = plt.Circle((self.l / 2, self.w / 2), 0.8, color=self.linecol, linewidth=linewidth)
        leftPenSpot = plt.Circle((11, self.w / 2), 0.8, color=self.linecol, linewidth=linewidth)
        rightPenSpot = plt.Circle((self.l - 11, self.w / 2), 0.8, color=self.linecol, linewidth=linewidth)

        # Draw Circles
        ax.add_patch(centreCircle)
        ax.add_patch(centreSpot)
        ax.add_patch(leftPenSpot)
        ax.add_patch(rightPenSpot)

        # Prepare Arcs (9.15m from penalty spot)
        leftArc = pat.Arc((11, self.w / 2), height=18.3, width=18.3, angle=0, theta1=308, theta2=52, color=self.linecol,
                          linewidth=linewidth)
        rightArc = pat.Arc((self.l - 11, self.w / 2), height=18.3, width=18.3, angle=0, theta1=128, theta2=232,
                           color=self.linecol,
                           linewidth=linewidth)
        bottomleftArc = pat.Arc((0, 0), height=3.5, width=3.5, angle=0, theta1=0, theta2=90, color=self.linecol,
                                linewidth=linewidth)
        topleftArc = pat.Arc((0, self.w), height=3.5, width=3.5, angle=0, theta1=270, theta2=360, color=self.linecol,
                             linewidth=linewidth)
        bottomrightArc = pat.Arc((self.l, 0), height=3.5, width=3.5, angle=0, theta1=90, theta2=180, color=self.linecol,
                                 linewidth=linewidth)
        toprightArc = pat.Arc((self.l, self.w), height=3.5, width=3.5, angle=0, theta1=180, theta2=270,
                              color=self.linecol,
                              linewidth=linewidth)

        # Draw Arcs
        ax.add_patch(leftArc)
        ax.add_patch(rightArc)
        ax.add_patch(bottomleftArc)
        ax.add_patch(topleftArc)
        ax.add_patch(bottomrightArc)
        ax.add_patch(toprightArc)

        # hide both axes (x and y) and all spines
        # equivalent to ax.axis('off') but that makes different coloring of ax and fig impossible!
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        return ax


# -----------------------------------------------
# create pitch as fig independent of class object
# -----------------------------------------------

def create_custom_Pitch(grasscol='#6aa84f', pitchcol='#38761d', linecol='white', figs=(10.5, 6.5),
                        l=105, w=65):
    # create the plt figure
    fig = plt.figure(figsize=figs)
    ax = fig.add_subplot(1, 1, 1)
    fig.set_facecolor(pitchcol)
    plt.xlim(0, l)
    plt.ylim(0, w)
    ax.patch.set_facecolor(grasscol)

    # Pitch Outline & Centre Line
    ax.plot([0, 0], [0, w], color=linecol, linewidth=5.5)
    ax.plot([0, l], [w, w], color=linecol, linewidth=5.5)
    ax.plot([l, l], [w, 0], color=linecol, linewidth=5.5)
    ax.plot([l, 0], [0, 0], color=linecol, linewidth=5.5)
    ax.plot([l / 2, l / 2], [0, w], color=linecol, linewidth=3.5)

    # Penalty areas - 7.32*0.5 + 5.5 + 11 = 20.16
    # Left
    ax.plot([16.5, 16.5], [w / 2 + 20.16, w / 2 - 20.16], color=linecol)
    ax.plot([0, 16.5], [w / 2 + 20.16, w / 2 + 20.16], color=linecol)
    ax.plot([16.5, 0], [w / 2 - 20.16, w / 2 - 20.16], color=linecol)
    # Right
    ax.plot([l, l - 16.5], [w / 2 + 20.16, w / 2 + 20.16], color=linecol)
    ax.plot([l - 16.5, l - 16.5], [w / 2 + 20.16, w / 2 - 20.16], color=linecol)
    ax.plot([l - 16.5, l], [w / 2 - 20.16, w / 2 - 20.16], color=linecol)

    # 6 yard boxes - 7.32*0.5 + 5.5 = 9.16
    # Left
    ax.plot([0, 5.5], [w / 2 + 9.16, w / 2 + 9.16], color=linecol)
    ax.plot([5.5, 5.5], [w / 2 + 9.16, w / 2 - 9.16], color=linecol)
    ax.plot([5.5, 0.5], [w / 2 - 9.16, w / 2 - 9.16], color=linecol)
    # Right
    ax.plot([l, l - 5.5], [w / 2 + 9.16, w / 2 + 9.16], color=linecol)
    ax.plot([l - 5.5, l - 5.5], [w / 2 + 9.16, w / 2 - 9.16], color=linecol)
    ax.plot([l - 5.5, l], [w / 2 - 9.16, w / 2 - 9.16], color=linecol)

    # Goals - 7.32*0.5
    ax.plot([l, l], [w / 2 - 3.66, w / 2 + 3.66], color=linecol, linewidth=9.75)
    ax.plot([0, 0], [w / 2 - 3.66, w / 2 + 3.66], color=linecol, linewidth=9.75)

    # Prepare Circles
    centreCircle = plt.Circle((l / 2, w / 2), 9.15, color=linecol, fill=False, linewidth=1.5)
    centreSpot = plt.Circle((l / 2, w / 2), 0.8, color=linecol)
    leftPenSpot = plt.Circle((11, w / 2), 0.8, color=linecol)
    rightPenSpot = plt.Circle((l - 11, w / 2), 0.8, color=linecol)

    # Draw Circles
    ax.add_patch(centreCircle)
    ax.add_patch(centreSpot)
    ax.add_patch(leftPenSpot)
    ax.add_patch(rightPenSpot)

    # Prepare Arcs (9.15m from penalty spot)
    leftArc = pat.Arc((11, w / 2), height=18.3, width=18.3, angle=0, theta1=308, theta2=52, color=linecol,
                      linewidth=1.5)
    rightArc = pat.Arc((l - 11, w / 2), height=18.3, width=18.3, angle=0, theta1=128, theta2=232, color=linecol,
                       linewidth=1.5)
    bottomleftArc = pat.Arc((0, 0), height=3.5, width=3.5, angle=0, theta1=0, theta2=90, color=linecol, linewidth=1.5)
    topleftArc = pat.Arc((0, w), height=3.5, width=3.5, angle=0, theta1=270, theta2=360, color=linecol, linewidth=1.5)
    bottomrightArc = pat.Arc((l, 0), height=3.5, width=3.5, angle=0, theta1=90, theta2=180, color=linecol,
                             linewidth=1.5)
    toprightArc = pat.Arc((l, w), height=3.5, width=3.5, angle=0, theta1=180, theta2=270, color=linecol, linewidth=1.5)

    # Draw Arcs
    ax.add_patch(leftArc)
    ax.add_patch(rightArc)
    ax.add_patch(bottomleftArc)
    ax.add_patch(topleftArc)
    ax.add_patch(bottomrightArc)
    ax.add_patch(toprightArc)

    # hide both axes (x and y) and all spines
    # equivalent to ax.axis('off') but that makes different coloring of ax and fig impossible!
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return fig
