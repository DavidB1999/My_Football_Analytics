# Radar charts

This directory contains 2 (3) options to create Radar charts. While radar charts undoubtedly have their drawbacks [1, 2, 3], I find them to be a very cool (and at some stage trendy :laughing:) way to visualize basic performance and compare players superficially. <br>

I myself created a very basic function that can be used to create an interactive radar charts. <br>
With the cool work done by other people, it seems somewhat pointless to spent hours and hours perfecting a function to do the same though.
I'd definitely recommend checking out the work from *mplsoccer* and *soccerplots* among others. <br>
Nevertheless, I adapted my favourite version of radar charts (as done my soccerplots) for my own use and made some minor adjustments. <br>

# Radar.py
**Radar.py** contains this adaptation along with my simplistic plotly-based function to create radar charts:

### param_select
                (df, params, var_col='Variables')

Function to select parameters from dataframe and **sort by the order of the parameters**!

**Parameters**

+ *df (pandas Dataframe)* - Dataframe to select from
+ *params (list, str)* - List of parameters to select
+ *var_col (str)* - Column in df where the params are to be found

**Returns**

*df_new (pandas Dataframe)*


### create_radar_plotly
                        (df1, player1, df2=None, player2=None, lines=False, var_col='Variables',
                        per_col='Percentiles', val_col='Values',
                        paper_bgcol='black', bgcol='white', ra_col='black', ra_gridcol='black',
                        ra_tickcol='black', aa_col='white', aa_gridcol='black', col1='red', col2='blue',
                        title_text='', title_col='white', title_size=20, legend=False,
                        legend_bgcol='black', legend_bordercol='white', legend_textcol='white')

Simplistic function to plot interactive radar chart with plotly.

**Parameters**

+ *df1, df2 (pandas Dataframe)* - Data for one or two players to be plotted
+ *player1, player2* - name(s) of the player(s)
+ *lines (boolean)* - Lines around the radar area(s)?
+ *var_col (str)* - Column in data where the params are to be found
+ *per_col (str)* - Column in data where the percentile ranks are to be found
+ *val_col (str)* - Column in data where the values are to be found
+ *paper_bgcol (str, color)* - Background color of the plot
+ *bgcol (str, color)* - Background color of the circular area
+ *ra_col, ra_gridcol, ra_tickcol (str, color)* - colors for the radial axes
+ *aa_col, aa_gridcol (str, color)* - colors for the angular axes
+ *col1, col2 (str, color)* - colors for the radars
+ *title_text (str)* - Title
+ *title_col (str, color), title_size (int)* - Title styling
+ *legend (boolean)* - Whether a legend is to be displayed
+ *legend_bgcol, legend_bordercol, legend_textcol (str, color)* - Legend styling

**Returns**

+ fig




## class Radar

**Attributes**

+ *background_col (str, color)* - defines the color of the plots background
+ *wedge_cols (list, str, color)* - 2 colors for the wedges forming the circular area of the plot; Recommendation: first color matching *background_col*
+ *radar_cols (list, str, color)* - 3 colors of which the first two form the polygon if one player is plotted. The last two are used for the two polygons if two players are plotted; Recommendation: first color matching the second *wedge_col* 
+ *fontfamily (str)* - Fontfamily for the entire plot
+ *label_fontsize (int), label_col (str, color)* - stylistics of labels indicating the variables around the chart
+ *range_fontsize (int), range_col (str, color)* - stylistics of labels indicating the range along the axis
+ *flip_labels (boolean)* - whether the labels on the lower half are supposed to flipped to increase readability 
+ *title_size (int), title_weight (str), title_col (str, color)* - stylistics of title on the left (player 1)
+ *subtitle_size (int), subtitle_weight (str), subtitle_col (str, color)* - stylistics of subtitle on the left (player 1)
+ *pos_title_size (int), pos_title_weight (str), pos_title_col (str, color)* - stylistics of sub-sub-title on the left (player 1)
+ *title_size_2 (int), title_weight_2 (str), title_col_2 (str, color)* - stylistics of title on the left (player 2)
+ *subtitle_size_2 (int), subtitle_weight_2 (str), subtitle_col_2 (str, color)* - stylistics of subtitle on the left (player 2)
+ *pos_title_size_2 (int), pos_title_weight_2 (str), pos_title_col_2 (str, color)* - stylistics of sub-sub-title on the left (player 2)
+ *endnote (str)* - text for the endnote at the bottom right
+ *endnote_size (int), endnote_weight (str), endnote_col (str, color)* - stylistics of the endnote
+ *y_endnote (float)* - y-coordinate of the endnote
+ *radii (list, float)* - radii for the wedges
+ *polygon_alpha (float)* - opacity for the polygons; Recommendation: <= 0.6


### plot_empty_radar
                    (self)

Function that plots only the empty chart; the foundation of the plot. <br>


### plot_radar
                (self, player1, p1_data, param_col='Variables', value_col='Values', percentile_col='Percentiles',
                ranges=None, params=None, title=None, subtitle=None, pos_title=None,
                title_2=None, subtitle_2=None, pos_title_2=None,
                endnote=None, player2=None, p2_data=None, display_type='Values')

**Parameters** 

+ *player1 (str)* - name of first player
+ *p1_data (pandas Dataframe)* - data for first player
+ *param_col (str)* - name of the column in the data containing the parameter names 
+ *value_col (str)* - name of the column in the data containing the values
+ *percentile_col (str)* - name of the column in the data containing the percentile ranks
+ *ranges (list, tuple, float)* - ranges for the plotted parameters 
+ *title, title_2(str)* title for player 1 (left) and title for player 2 (right)
+ *subtitle, subtitle_2(str)* subtitle for player 1 (left) and subtitle for player 2 (right)
+ *pos_title, pos_title_2(str)* sub-sub-title for player 1 (left) and sub-sub-title for player 2 (right)
+ *endnote (str)* - Text for the endnote (overwrites endnote text defined in class)
+ *player2 (str)* - name of second player
+ *p2_data (pandas Dataframe)* - data for second player; optional and if supplied both players will be plotted
+ *display_types (str)* - Either "Values" or "Percentile" - whether values or percentiles are to be displayed

**Returns**

+ ax


### add_labels
            (self, params, ax, radius, return_xy, labelsnotvalues=True):

Function to plot labels for the radar charts. Is used in *plot_radar* to plot both the parameter-labels around the circular area and the value along the axis.<br>

**Parameters** 

+ *params (list)* - A list of either the parameter names or the values to be plotted for each parameter at a given radius
+ *ax (ax)* - ax to plot to
+ *radius (float)* - Radius to which the labels apply
+ *return_xy (boolean)* - decided whether the calculated coordinates of the labels' positions are returned (needed for values but not for labels)
+ *labelsnotvalues (boolean)* - parameter-labels around the circular area or the values along the axis?

**Returns**

+ ax


### add_value_ranges
                    (self, ranges, ax)

Function to plot all the value ranges on the radial axes. Calls *add_labels* for every radius in *radii*.

**Parameters** 

+ *ranges (list, tuple, float)* - ranges for the plotted parameters 
+ *ax (ax)* - ax to plot to

**Returns**

+ ax


### plot_wedges
                (self, ax)

Function to plot the wedges forming the circular area. This is where my implementation deviates from soccerplot's. 
I use wedges instead of circles and use an extra function for the plotting of polygons. <br>

**Parameters**

+ *ax (ax)* - ax to plot to

**Returns**

+ ax


### plot_polygons
                (self, ax, vertices, num_players=1)

Function to plot the polygons.
If just one player is plotted the polygon is hidden behind the wedges but additional wedges are plotted and visible only where the polygon is.
If two players are plotted 2 polygons are plotted on top of the wedges from *plot_wedges*. <br>

**Parameters** 

+ *ax (ax)* - ax to plot to
+ *vertices (ndim list, float) - list with lists with x and y coordinates for each parameter's corner in the polygon
+ *num_players (int) - number of players plotted (1 or 2)

**Returns**

+ ax


# radar_utils.py


## Credits

*McKay Johns* with a quick and simple introduction into *soccerplots* radar charts - https://youtu.be/cXtC2EOQj6Q <br>
*mplsoccer* - https://mplsoccer.readthedocs.io/en/latest/gallery/radar/plot_radar.html <br>
*soccerplots* - https://github.com/Slothfulwave612/soccerplots/blob/master/docs/radar_chart.md#changing-alpha-values-for-comparison-radar <br>

## References

[1] https://www.storytellingwithdata.com/blog/2021/8/31/what-is-a-spider-chart <br>
[2] https://blog.scottlogic.com/2011/09/23/a-critique-of-radar-charts.html <br>
[3] https://statsbomb.com/articles/soccer/revisiting-radars/
