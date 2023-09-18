# My own function to create a football pitch

# -------------------------
# import necessary packages
# -------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat


# pitch dimensions based on https://de.wikipedia.org/wiki/Fu%C3%9Fballregeln in meters
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
    plt.show()

