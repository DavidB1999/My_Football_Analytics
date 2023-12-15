import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly
import radar_utils as ru
from matplotlib.patches import Polygon
from matplotlib import patches


# -------------------------------------------------------------------
# function to select parameters of interest and filter df accordingly
# -------------------------------------------------------------------

def param_select(df, params, var_col='Variables'):
    df_new = df[df[var_col].isin(params)]  # select columns of interest

    # merge with column of parameters in correct order
    # to ensure data is sorted by the order of parameters selected
    dummy = pd.Series(params, name=var_col).to_frame()
    df_new = pd.merge(dummy, df_new, on=var_col, how='left')

    return df_new


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


class Radar:

    # class with all necessary methods
    def __init__(self,
                 background_col='#1f142a', wedge_cols=['#1f142a', '#675E71'], radar_cols=['#675E71', 'green'],
                 fontfamily='Liberation Serif', label_fontsize=10, label_col='#efeef0',
                 range_fontsize=8, range_col='#efeef0',
                 title_size=15, title_weight='bold', title_col='#274e13',
                 subtitle_size=12, subtitle_weight='bold', subtitle_col='#274e13',
                 pos_title_size=10, pos_title_weight='regular', pos_title_col='#274e13',
                 title_size_2=15, title_weight_2='bold', title_col_2='#4e1327',
                 subtitle_size_2=12, subtitle_weight_2='bold', subtitle_col_2='#4e1327',
                 pos_title_size_2=10, pos_title_weight_2='regular', pos_title_col_2='#4e1327',
                 endnote='Inspired by Statsbomb | Adapted from soccerplots',
                 endnote_size=10, endnote_weight='regular', endnote_col='#efeef0', y_endnote=-13,
                 radii=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], polygon_alpha=0.6):

        self.background_col = background_col
        self.wedge_cols = wedge_cols
        self.radar_cols = radar_cols
        self.ff = fontfamily
        self.lbfs = label_fontsize
        self.lbcol = label_col
        self.rgfs = range_fontsize
        self.rgcol = range_col
        self.ts = title_size
        self.tw = title_weight
        self.tcol = title_col
        self.sts = subtitle_size
        self.stw = subtitle_weight
        self.stcol = subtitle_col
        self.pts = pos_title_size
        self.ptw = pos_title_weight
        self.ptcol = pos_title_col
        self.ts2 = title_size_2
        self.tw2 = title_weight_2
        self.tcol2 = title_col_2
        self.sts2 = subtitle_size_2
        self.stw2 = subtitle_weight_2
        self.stcol2 = subtitle_col_2
        self.pts2 = pos_title_size_2
        self.ptw2 = pos_title_weight_2
        self.ptcol2 = pos_title_col_2
        self.endnote = endnote
        self.es = endnote_size
        self.ew = endnote_weight
        self.ecol = endnote_col
        self.y_end = y_endnote
        self.radii = radii
        self.pol_al = polygon_alpha

    def plot_empty_radar(self):

        # set up figure
        fig, ax = plt.subplots(figsize=(20, 10), facecolor=self.background_col)
        ax.set_facecolor(self.background_col)

        # set axis
        ax.set_aspect('equal')
        ax.set(xlim=(-15, 15), ylim=(-15, 15))

        # max radius (i.e. outermost wedge)
        radius = self.radii[-1]
        ax = self.plot_wedges(ax=ax)

        if self.endnote:
            y_end = self.y_end
            for note in self.endnote.split('\n'):
                ax.text(14.5, y_end, note, ha='right', fontdict={"color": self.ecol}, fontsize=self.es)
                y_end -= 0.75

        ax.axis('off')

        return ax



    ### IDEA:
    ### One function for overall wedges
    ### Another function for polygons and the second wedge layer!
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

    def plot_polygons(self, ax, vertices, num_player, alpha):

        # either all players in one go loop over players and run this function for each player
        # handle color accordingly

        zow = 2
        radar = Polygon(vertices, fc=self.radar_cols[num_player - 1], alpha=self.pol_al)
        ax.add_patch(radar)
        for rad in self.radii:
            if rad % 2 == 0:
                wedge = patches.Wedge(center=(0, 0), r=rad + 1, theta1=0, theta2=360, width=1, color=self.radar_cols[1],
                                      zorder=zow)
                wedge.set_clip_path(radar)
                ax.add_patch(wedge)
            else:
                wedge = patches.Wedge(center=(0, 0), r=rad + 1, theta1=0, theta2=360, width=1, color=self.radar_cols[0],
                                      zorder=zow)
                wedge.set_clip_path(radar)
                ax.add_patch(wedge)

        return ax

    def plot_radar(self, player1, p1_data,
                   ranges, params, values, radar_color, alphas, title=dict(), compare=False,
                   endnote=None, end_size=9, end_color='#efeef0',
                   player2=None, p2_data=None, ):

        # assert that name and data for second player are only supplied in combination
        if player2:
            assert p2_data is not None, "If you supply a name for a second player you need to supply data as well"
        elif p2_data:
            raise Warning("If you supply data for a second player, I'd recommend giving him a name.")

        # assert required conditions
        assert len(ranges) >= 3, "Length of ranges should be greater than equal to 3"
        assert len(params) >= 3, "Length of params should be greater than equal to 3"

        if p2_data:
            ## for making comparison radar charts
            assert len(values) == len(radar_color) == len(
                alphas), "Length for values, radar_color and alpha do not match"
        else:
            assert len(values) >= 3, "Length of values should be greater than equal to 3"
            assert len(ranges) == len(params) == len(values), "Length for ranges, params and values not match"

        fig, ax = plt.subplots(figsize=(20, 10), facecolor=self.background_col)
        ax.set_facecolor(self.background_col)

        # what is the point?!
        ## set axis
        # ax.set_aspect('equal')
        # ax.set(xlim=(-22, 22), ylim=(-23, 25))

        if type(radar_color) == str:
            ## make radar_color a list
            radar_color = [radar_color]
            radar_color.append('#D6D6D6')  # light grey
