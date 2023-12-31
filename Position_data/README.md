# Position data

This directory is supposed to help handle position data / tracking data from different data sources.
To this purpose, a class tracking_data was written, containing a collection of functions to use with and
around tracking data. <br>

## class tracking_data

**Attributes**

+ *data (dataframe)*: dataframe containing tracking data in wide format (one column for x and y for each player)
+ *data_source (str)* - data origin (influences data handling)
+ *x_range_data* and *y_range_data (tuple, numeric)* - range of x and y coordinates in the *shot_data*
+ *x_range_pitch* and *x_range_pitch (tuple, numeric)* - range of x and y coordinates required; data will be scaled accordingly
+ *mirror_away (list, str)* - list of axes (x, y) to mirror for the away team 
+ *x_cols_pattern* and *y_cols_pattern (str)* - pattern to look for to identify columns with the coordinates; also used for naming of produced data
+ *scale_to_pitch (str)* - default options of data ranges to scale to
+ *mirror_second_half (boolean)* - indicating whether data from second half is supposed to be mirrored
+ *home* and *away (str)* - optional names for both teams
+ *period_col (str)* - column indicating the period of play / half
+ *time_col (str)* - column with the time
+ *fps (int)* - frame rate / frames per second of the data
+ *got_velocities (booean)* - used to store information on whether get_velocities() has already been called


### rescale_tracking_data
                        (self)

Function to initially handle tracking data. The main purpose is the rescaling of x and y coordinates to match to a certain
data range as for instance necessary for pitches. An indepth explanation of the rescaling logic used can be found
in the Appendix. <br>

**Parameters** 

**Returns** 

+ data (dataframe) - rescaled shot_data


### plot_players
                (self, frame, pitch_col='#1c380e', line_col='white', colors=['red', 'blue', 'black'],
                     velocities=False, PlayerAlpha=0.7)
Function to plot all players on a pitch. This functions is mostly based on Laurie Shaw's function to plot players in *Metrica_Viz.py*. <br>

**Parameters**

+ *frame (int)* - number of the frame to be plotted
+ *pitch_col (color)* - color of the pitch
+ *line_col (color)* - color of the pitch lines
+ *colors (list, color)* - list of colors for home team, away team and the ball (in that order)
+ *velocities (boolean)* - Whether velocities are supposed to be displayed
+ *PlayerAlpha (float)* - Opacity/alpha of velocity quiver

**Returns**

+ *fig (figure)* - Plot of players on a pitch


### get_velocities
                (self, data=None, smoothing=True, filter='Savitzky-Golay', window=7, polyorder=1, maxspeed=12)

Function to get players' velocities from Laurie Shaw's *Metrica_Velocities.py*. Minor adaptions have been made. <br>

**Parameters**

+ *data (dataframe)* - Dataframe with tracking data; If None the class' data will be used. 
+ *smoothing (boolean)* - Whether data should be smoothed (recommended)
+ *filter (str)* - The smoothing filter to be used of smoothing is True
+ *window (int)* - The number of frames to be included in the filter windows
+ *polyorder (int)* - polyorder for Savitzky-Golay filter
+ *max_speed (float)* - maximal speed deemed realistic to remove erroneous velocities

**Returns**

+ *data (dataframe)* - the data with added columns of velocity


### remove_velocities
                    (self, data)
Function to remove velocities from data from Laurie Shaw. <br>

**Parameters**

+ *data (dataframe)* - Dataframe with tracking data 

**Returns**

+ *data (dataframe)* - the data with any velocity columns removed


### animation_clip
                    (self, frames_per_second=25, fname='Animated_Clip',pitch_col='#1c380e',
                    line_col='white', data=None, frames=None, colors= ['red', 'blue', 'black'],
                    velocities=False, PlayerAlpha=0.7, fpath=None)

Function to animate a set of frames of plotted players à la *plot_players*. 
Once again this is largely based on Laurie Shaw's code in *Metrica_Viz.py*. <br>


**Parameters**

+ *frames_per_second (int)* - frames per second to assume when generating the movie. Default is 25.
+ *fname (str)* - intended file name to store the clip. Defaults to "Animated_Clip"
+ *pitch_col (color)* - color of the pitch
+ *line_col (color)* - color of the pitch lines
+ *data (dataframe)* - Dataframe with tracking data; If None the class' data will be used. 
+ *frames (int, tuple)* - range of frames to be included in the clip. If None, the entire data will be used - not recommended.
+ *colors (list, color)* - list of colors for home team, away team and the ball (in that order)
+ *velocities (boolean)* - Whether velocities are supposed to be displayed
+ *PlayerAlpha (float)* - Opacity/alpha of velocity quiver
+ *fpath (str)* - Directory to save the Clip. If None Clip will be stored in current directory.


**Returns**



## Credits

Laurie Shaw - https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking/blob/master/Metrica_Velocities.py |
https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking/blob/master/Metrica_Viz.py


## Appendix

**Rescaling logic:** <br>

$$ shotdataminmax = (a_1, a_2)$$

$$ pitchminmax = (b_1, b_2)$$

$$ \Delta a = a_2- a_1$$ 

$$ \Delta b = b_2- b_1$$ 

$$ c = coordinate$$

$$ scalingfactor = s = \frac{\Delta b}{\Delta a}$$

**Without mirroring:** <br>

$$ c_{new}  = b_1 + (c + a_1\times (-1))\times s$$

or 

$$ c_{new}  = b_2 + (c + a_2\times (-1))\times s$$

**With mirroring:** <br>

$$ c_{new}  = b_2 - (c + a_1\times (-1))\times s$$

or 

$$ c_{new}  = b_1 -1 (c + a_2\times (-1))\times s$$

To mirror after data has already been scaled just use 1 for s! <br>


For more details check [Scaling_Reasoning.xlsx](../ShotData/Scaling_Reasoning.xlsx). <br>
