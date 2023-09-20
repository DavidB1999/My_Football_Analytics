import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly


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