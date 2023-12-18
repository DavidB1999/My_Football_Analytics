import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly
import radar_utils as ru
from matplotlib.patches import Polygon
from matplotlib import patches
import warnings


# ------------------------------------------------------
# function to create interactive radar chart with plotly
# ------------------------------------------------------

def create_radar_plotly(df1, player1, df2=None, player2=None, lines=False, var_col='Variables',
                        per_col='Percentiles', val_col='Values',
                        paper_bgcol='black', bgcol='white', ra_col='black', ra_gridcol='black',
                        ra_tickcol='black', aa_col='white', aa_gridcol='black', col1='red', col2='blue',
                        title_text='', title_col='white', title_size=20, legend=False,
                        legend_bgcol='black', legend_bordercol='white', legend_textcol='white'):
    # define data for the radar chart
    r1 = [int(x) for x in df1[per_col]]  # percentile values (need to be available or calculated before!)
    values1 = [float(x) for x in df1[val_col]]  # the actual numeric values
    parameters1 = df1[var_col].tolist()

    # duplicate the first element for closing line (only relevant if lines are drawn!)
    if lines:
        r1.append(r1[0])
        values1.append(values1[0])
        parameters1.append(parameters1[0])

    if df2 is not None:
        r2 = [int(x) for x in df2[per_col]]
        values2 = [float(x) for x in df2[val_col]]
        parameters2 = df2[var_col].tolist()

        # duplicate the first element for closing line (only relevant if lines are drawn!)
        if lines:
            r2.append(r2[0])
            values2.append(values2[0])
            parameters2.append(parameters2[0])

        # parameters for both player should be identical
        if parameters1 != parameters2:
            raise Exception('Please make sure the parameters in both dataframes are identical!')
        else:
            parameters = parameters1
    else:
        parameters = parameters1

    fig = go.Figure()

    # player 1 trace
    fig.add_trace(go.Scatterpolar(
        r=r1,
        theta=parameters,
        fill='toself',
        name=player1,
        mode='markers',
        customdata=values1,
        hovertemplate=(
                'Percentile: %{r} <br>' +
                'Value: %{customdata}'
        )))

    # player 2 trace
    if df2 is not None:
        fig.add_trace(go.Scatterpolar(
            r=r2,
            theta=parameters,
            fill='toself',
            name=player2,
            customdata=values2,
            mode='markers',
            hovertemplate=(
                    'Percentile: %{r} <br>' +
                    'Value: %{customdata}'
            )))

    fig.update_layout(
        # general visual
        paper_bgcolor=paper_bgcol,
        # polar area
        polar=dict(
            hole=0,
            bgcolor=bgcol,
            gridshape='circular',
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                color=ra_col,
                gridcolor=ra_gridcol,
                showline=False,
                tickfont=dict(
                    color=ra_tickcol,
                    size=10)),
            angularaxis=dict(
                color=aa_col,
                gridcolor=aa_gridcol,
                layer='below traces'
            )
        ),

        showlegend=legend,
        colorway=[col1, col2],  # colors for the traces

        # hoverlabel
        hoverlabel=dict(
            # bgcolor='pink',
            bordercolor='black',
            font=dict(
                color='white'
            )
        ),

        # title
        title=dict(
            text=title_text,
            font=dict(
                size=title_size,
                color=title_col),
            x=0.5, xanchor='center',
            y=.975, yanchor='top'
        ),

        # legend
        legend=dict(
            bgcolor=legend_bgcol,
            bordercolor=legend_bordercol,
            font=dict(
                size=12,
                color=legend_textcol),
            title=dict(
                text='Players',
                font=dict(
                    size=15)
            )
        )
    )
    return fig


# --------------------------------------------------------------
# class for custom radar charts based on soccerplots radar class
# --------------------------------------------------------------

class Radar:

    # class with all necessary methods
    def __init__(self,
                 background_col=None, wedge_cols=None,radar_cols=None, fontfamily=None,
                 label_fontsize=None, label_col=None, range_fontsize=None, range_col=None,
                 flip_labels=None, title_size=None, title_weight=None, title_col=None,
                 subtitle_size=None, subtitle_weight=None, subtitle_col=None,
                 pos_title_size=None, pos_title_weight=None, pos_title_col=None,
                 title_size_2=None, title_weight_2=None, title_col_2=None,
                 subtitle_size_2=None, subtitle_weight_2=None, subtitle_col_2=None,
                 pos_title_size_2=None, pos_title_weight_2=None, pos_title_col_2=None,
                 endnote=None, endnote_size=None, endnote_weight=None, endnote_col=None, y_endnote=None,
                 radii=None, polygon_alpha=None, radar_cols_to_title=False,
                 style='moody'):

        if style == 'moody':
            self.background_col = background_col if background_col else '#1f142a'
            self.wedge_cols = wedge_cols if wedge_cols else ['#1f142a', '#675E71']
            self.radar_cols = radar_cols if radar_cols else ['#675E71', '#99d8bc', '#d899b5']
            self.ff = fontfamily if fontfamily else 'Liberation Serif'
            self.lbfs = label_fontsize if label_fontsize else 10
            self.lbcol = label_col if label_col else '#efeef0'
            self.rgfs = range_fontsize if range_fontsize else 8
            self.rgcol = range_col if range_col else '#efeef0'
            self.flip_labels = flip_labels if flip_labels else True
            self.ts = title_size if title_size else 15
            self.tw = title_weight if title_weight else'bold'
            self.tcol = title_col if title_col else '#99d8bc'
            self.sts = subtitle_size if subtitle_size else 12
            self.stw = subtitle_weight if subtitle_weight else 'bold'
            self.stcol = subtitle_col if subtitle_col else '#99d8bc'
            self.pts = pos_title_size if pos_title_size else 10
            self.ptw = pos_title_weight if pos_title_weight else 'regular'
            self.ptcol = pos_title_col if pos_title_col else '#99d8bc'
            self.ts2 = title_size_2 if title_size else 15
            self.tw2 = title_weight_2 if title_weight else'bold'
            self.tcol2 = title_col_2 if title_col else '#d899b5'
            self.sts2 = subtitle_size_2 if subtitle_size else 12
            self.stw2 = subtitle_weight_2 if subtitle_weight else 'bold'
            self.stcol2 = subtitle_col_2 if subtitle_col else '#d899b5'
            self.pts2 = pos_title_size_2 if pos_title_size else 10
            self.ptw2 = pos_title_weight_2 if pos_title_weight else 'regular'
            self.ptcol2 = pos_title_col_2 if pos_title_col else '#d899b5'
            self.endnote = endnote if endnote else 'Inspired by Statsbomb | Adapted from soccerplots'
            self.es = endnote_size if endnote_size else 10
            self.ew = endnote_weight if endnote_weight else 'regular'
            self.ecol = endnote_col if endnote_col else '#efeef0'
            self.y_end = y_endnote if y_endnote else -13.5
            self.radii = radii if radii else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            self.radius = max(self.radii)
            self.pol_al = polygon_alpha if polygon_alpha else 0.6
        elif style == 'light':
            self.background_col = background_col if background_col else '#fcfdfb'
            self.wedge_cols = wedge_cols if wedge_cols else ['#fcfdfb', '#fff9b5']
            self.radar_cols = radar_cols if radar_cols else ['#fff9b5', '#3dd5f0', '#e45c5e']
            self.ff = fontfamily if fontfamily else 'Liberation Serif'
            self.lbfs = label_fontsize if label_fontsize else 10
            self.lbcol = label_col if label_col else '#000000'
            self.rgfs = range_fontsize if range_fontsize else 8
            self.rgcol = range_col if range_col else '#000000'
            self.flip_labels = flip_labels if flip_labels else True
            self.ts = title_size if title_size else 15
            self.tw = title_weight if title_weight else'bold'
            self.tcol = title_col if title_col else '#3dd5f0'
            self.sts = subtitle_size if subtitle_size else 12
            self.stw = subtitle_weight if subtitle_weight else 'bold'
            self.stcol = subtitle_col if subtitle_col else '#3dd5f0'
            self.pts = pos_title_size if pos_title_size else 10
            self.ptw = pos_title_weight if pos_title_weight else 'regular'
            self.ptcol = pos_title_col if pos_title_col else '#3dd5f0'
            self.ts2 = title_size_2 if title_size else 15
            self.tw2 = title_weight_2 if title_weight else'bold'
            self.tcol2 = title_col_2 if title_col else '#e45c5e'
            self.sts2 = subtitle_size_2 if subtitle_size else 12
            self.stw2 = subtitle_weight_2 if subtitle_weight else 'bold'
            self.stcol2 = subtitle_col_2 if subtitle_col else '#e45c5e'
            self.pts2 = pos_title_size_2 if pos_title_size else 10
            self.ptw2 = pos_title_weight_2 if pos_title_weight else 'regular'
            self.ptcol2 = pos_title_col_2 if pos_title_col else '#e45c5e'
            self.endnote = endnote if endnote else 'Inspired by Statsbomb | Adapted from soccerplots'
            self.es = endnote_size if endnote_size else 10
            self.ew = endnote_weight if endnote_weight else 'regular'
            self.ecol = endnote_col if endnote_col else '#000000'
            self.y_end = y_endnote if y_endnote else -13.5
            self.radii = radii if radii else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            self.radius = max(self.radii)
            self.pol_al = polygon_alpha if polygon_alpha else 0.6
        if radar_cols_to_title:
            self.tcol = radar_cols[1]
            self.stcol = radar_cols[1]
            self.ptcol = radar_cols[1]
            self.tcol2 = radar_cols[2]
            self.stcol2 = radar_cols[2]
            self.ptcol2 = radar_cols[2]
    # ---------------------------------------------------------
    # function to plot empty radar chart area (only the wedges)
    # ---------------------------------------------------------

    def plot_empty_radar(self, y_endnote=None, endnote=None):

        # optional to overwrite endnote settings
        if endnote:
            pass
        else:
            endnote = self.endnote

        if y_endnote:
            pass
        else:
            y_endnote = self.y_end

        # set up figure
        fig, ax = plt.subplots(figsize=(20, 10), facecolor=self.background_col)
        ax.set_facecolor(self.background_col)

        # set axis
        ax.set_aspect('equal')
        ax.set(xlim=(-15, 15), ylim=(-15, 15))

        # max radius (i.e. outermost wedge)
        radius = self.radii[-1]
        ax = self.plot_wedges(ax=ax)

        if endnote:
            for note in endnote.split('\n'):
                ax.text(14.5, y_endnote, note, ha='right', fontdict={"color": self.ecol}, fontsize=self.es)
                y_endnote -= 0.75

        ax.axis('off')

        return ax

    # ----------------------------------------
    # function to plot the actual radar chart
    # ----------------------------------------

    def plot_radar(self, player1, p1_data, param_col='Variables', value_col='Values', percentile_col='Percentiles',
                   ranges=None, params=None, title=None, subtitle=None, pos_title=None,
                   title_2=None, subtitle_2=None, pos_title_2=None, y_endnote=None,
                   endnote=None, player2=None, p2_data=None, display_type='Values'):

        # optional to overwrite endnote settings
        if endnote:
            pass
        else:
            endnote = self.endnote

        if y_endnote:
            pass
        else:
            y_endnote = self.y_end

        # assert that name and data for second player are only supplied in combination
        if player2:
            assert p2_data is not None, "If you supply a name for a second player you need to supply data as well"
            n_players = 2
        elif p2_data is not None:
            warnings.warn("If you supply data for a second player, I'd recommend giving him a name.")
            n_players = 2
        else:
            n_players = 1

        # get the data
        if params:
            warnings.warn("By supplying your own parameter names, a match between those and the values cannot be "
                          "assured!")
            params_supplied = True
        else:
            params = p1_data[param_col]
            params_supplied = False
        org_values1 = p1_data[value_col]
        percentiles1 = p1_data[percentile_col]
        if p2_data is not None:
            org_values2 = p2_data[value_col]
            percentiles2 = p2_data[percentile_col]
            if params_supplied:
                pass
            else:
                assert all(params == p2_data[param_col]), 'Parameters for both players need to match! You can ' \
                                                          'circumvent this problem by supplying a list of parameters,' \
                                                          ' but this is not recommended! '

        # ranges either by method or supplied!
        if ranges:
            pass
        elif display_type != 'Percentile':
            raise ValueError('You need to supply ranges! A method to create them automatically is not yet included!'
                             'If you you used data from FBref for the radar chart you might want to consider looking '
                             'for per value ranges on their website.')

        if display_type == 'Percentile':
            values1 = percentiles1
            if p2_data is not None:
                values2 = percentiles2
            if ranges:
                pass
            else:
                ranges = [(0, 100) for p in params]
        else:
            values1 = org_values1
            if p2_data is not None:
                values2 = org_values2

        # assert required conditions
        assert len(ranges) >= 3, "Length of ranges should be greater than equal to 3"
        assert len(params) >= 3, "Length of params should be greater than equal to 3"
        assert len(params) == len(ranges) == len(values1) == len(percentiles1), "Number of parameters, ranges, " \
                                                                                "values and percentiles ranks " \
                                                                                "must match!"
        if p2_data is not None:
            assert len(values1) == len(values2) == len(percentiles2), "The number of values and percentile " \
                                                                      "ranks for both players must match! "

        fig, ax = plt.subplots(figsize=(20, 10), facecolor=self.background_col)
        ax.set_facecolor(self.background_col)

        ax.set_aspect('equal')
        ax.set(xlim=(-15, 15), ylim=(-15, 15))

        rotations = ru.get_label_coordinates(n=len(params))[:, 2]

        if p2_data is not None:
            vertices1 = ru.get_radar_coord(values=values1, radius=self.radius, ranges=ranges, rot=rotations)
            vertices2 = ru.get_radar_coord(values=values2, radius=self.radius, ranges=ranges, rot=rotations)
            vertices = [vertices1, vertices2]
        else:
            vertices = ru.get_radar_coord(values=values1, radius=self.radius, ranges=ranges, rot=rotations)

        ax = self.plot_wedges(ax=ax)
        ax = self.plot_polygons(ax=ax, vertices=vertices, num_players=n_players)
        ax = self.add_labels(params=params, ax=ax, return_xy=False, radius=self.radius + 2)
        ax, xy, range_values = self.add_value_ranges(ranges=ranges, ax=ax)

        if title:
            ax.text(-14.5, 14.5, title, ha='left', va='top',
                    fontdict={'color': self.tcol, 'size': self.ts, 'weight': self.tw})
        if title_2:
            ax.text(14.5, 14.5, title_2, ha='right', va='top',
                    fontdict={'color': self.tcol2, 'size': self.ts2, 'weight': self.tw2})
        if subtitle:
            ax.text(-14.5, 13.5, subtitle, ha='left', va='top',
                    fontdict={'color': self.stcol, 'size': self.sts, 'weight': self.stw})
        if subtitle_2:
            ax.text(14.5, 13.5, subtitle_2, ha='right', va='top',
                    fontdict={'color': self.stcol2, 'size': self.sts2, 'weight': self.stw2})
        if pos_title:
            ax.text(-14.5, 12.5, pos_title, ha='left', va='top',
                    fontdict={'color': self.ptcol, 'size': self.pts, 'weight': self.ptw})
        if pos_title_2:
            ax.text(14.5, 12.5, pos_title_2, ha='right', va='top',
                    fontdict={'color': self.ptcol2, 'size': self.pts2, 'weight': self.ptw2})

        if endnote:
            for note in endnote.split('\n'):
                ax.text(14.5, y_endnote, note, ha='right', fontdict={"color": self.ecol}, fontsize=self.es)
                y_endnote -= 0.75

        ax.axis('off')

        return ax

    def add_labels(self, params, ax, radius, return_xy, labelsnotvalues=True):
        # radius = max[self.radii] + 2
        coord = ru.get_label_coordinates(n=len(params))

        xy = []
        for i, p in enumerate(params):
            rot = coord[i, 2]
            x, y = (radius * np.cos(rot), radius * np.sin(rot))
            xy.append((x, y))

            # adding 180° if y < 0 => on its head
            if y < 0 and self.flip_labels:
                rot += np.pi

            # if parameters is numeric (i.e. value instead if label) round
            if type(p) == np.float64:
                p = round(p, 2)
            else:
                pass

            # ensure correct styling dependent on whether values or labels are plotted
            if labelsnotvalues:
                size = self.lbfs
                col = self.lbcol
            else:
                size = self.rgfs
                col = self.rgcol

            # add text to ax
            ax.text(x, y, p, rotation=np.rad2deg(rot) - 90, ha='center', va='center',
                    fontsize=size, fontdict=dict(color=col))

        if return_xy:
            return ax, xy
        else:
            return ax

    def add_value_ranges(self, ranges, ax):
        xys = []
        n_range_values = len(self.radii)
        range_values = np.array([])
        # looping over list of tuples of min and max for each param
        # get num values evenly spread from min to max
        for rng in ranges:
            value = np.linspace(start=rng[0], stop=rng[1], num=n_range_values)
            range_values = np.append(range_values, value)

        # reshape to number of params (rows) * num values (columns)
        range_values = range_values.reshape((len(ranges), n_range_values))

        # loop over radius
        for i, r in enumerate(self.radii):
            # get the value of each parameter that belongs to this radius
            values = range_values[:, i]

            # for these values and this radius create the labels
            # get both the ax and the coordinates of the value labels
            ax, xy = self.add_labels(params=values, ax=ax, return_xy=True, radius=r + 1, labelsnotvalues=False)
            xys.append(xy)

        return ax, np.array(xys), range_values


    def plot_wedges(self, ax):

        # zorder for wedges
        zow = 2
        for rad in self.radii:
            if rad % 2 == 0:
                wedge = patches.Wedge(center=(0, 0), r=rad + 1, theta1=0, theta2=360, width=1, color=self.wedge_cols[1],
                                      zorder=zow)
                ax.add_patch(wedge)
            else:
                wedge = patches.Wedge(center=(0, 0), r=rad + 1, theta1=0, theta2=360, width=1, color=self.wedge_cols[0],
                                      zorder=zow)
                ax.add_patch(wedge)

        return ax

    def plot_polygons(self, ax, vertices, num_players=1):  # alpha

        # either all players in one go loop over players and run this function for each player
        # handle color accordingly
        zow = 2

        if num_players == 1:
            radar = Polygon(vertices, fc=self.radar_cols[0], alpha=self.pol_al)
            ax.add_patch(radar)
            for rad in self.radii:
                if rad % 2 == 0:
                    wedge = patches.Wedge(center=(0, 0), r=rad + 1, theta1=0, theta2=360, width=1,
                                          color=self.radar_cols[1], zorder=zow)
                    wedge.set_clip_path(radar)
                    ax.add_patch(wedge)
                else:
                    wedge = patches.Wedge(center=(0, 0), r=rad + 1, theta1=0, theta2=360, width=1,
                                          color=self.radar_cols[0], zorder=zow)
                    wedge.set_clip_path(radar)
                    ax.add_patch(wedge)
        else:
            # split vertices by player
            v1 = vertices[0]
            v2 = vertices[1]

            if self.pol_al > 0.6:
                warnings.warn('With more than one player plotted an alpha < 0.6 is recommended!')
            # create and add radar for both players
            radar1 = Polygon(v1, fc=self.radar_cols[1], alpha=self.pol_al, zorder=zow + 1)
            ax.add_patch(radar1)
            radar2 = Polygon(v2, fc=self.radar_cols[2], alpha=self.pol_al, zorder=zow + 1)
            ax.add_patch(radar2)

        return ax
