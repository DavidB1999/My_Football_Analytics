# Event Data

Module for comfortable handling of event data. Supported data sources so far: 'Statsbomb' [1, 2] & 'metrica' [3]. <br>
Credit to Laurie Shaw's module on FriendsofTracking and Statsbomb's GitHub!


### def load_metrica_event_data
                            (datadir, fname)

Simple function to load *metrica* event data from local directory.

**Parameters**

+ *datadir (str)* - file directory (if not current working directory)
+ *fname (str)* file name of event data file

**Returns** 

+ *events (pd.DataFrame)* - event data as pandas DataFrame


## class event_data

**Attributes**

+ *data (dataframe)*: dataframe containing event data
+ *data_source (str)* - data origin (influences data handling)
+ *x_range_data* and *y_range_data (tuple, numeric)* - range of x and y coordinates in the *shot_data*
+ *x_range_pitch* and *x_range_pitch (tuple, numeric)* - range of x and y coordinates required; data will be scaled accordingly
+ *scale_to_pitch (str)* - default options of data ranges to scale to
+ *mirror_second_half (boolean)* - indicating whether data from second half is supposed to be mirrored
+ *fps (int)* - frame rate / frames per second of the data
+ *mirror_away (list, str)* - list of axes (x, y) to mirror for the away team 
+ *colors (list, colors)* - colors for home and away team and the ball and events
+ *home_team* & *away_team (str)* - optional names of home and away teams; has to match the names uses in original data!

## rescale_event_data 
                    (self)
 
Function called on initiation of class object, rescaling the data according to the logic depicted in appendix.
The function also restructures the data for all data sources to a common format.
For 'Statsbomb' data this entails (for now) the removing of lots of variables. <br> 

**Returns**

+ *new_data (pd.DataFrame)* - rescaled and reformatted data


## get_event_by_type
                    (self, event_type)

Function to filter the class data by event type.

**Parameters**

+ *event_type (str)* - Event type of interest. String must match exactly with the name of the event type.

**Returns** 

+ *ev_da (pd.DataFrame)* - filtered event data as pandas DataFrame


## Credits

Laurie Shaw - https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking/blob/master/Metrica_Velocities.py |
https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking/blob/master/Metrica_Viz.py <br>
Statsbomb -  https://statsbomb.com/what-we-do/hub/free-data/ | https://github.com/metrica-sports/sample-data

## References

[1] - https://github.com/statsbomb/statsbombpy <br>
[2] - https://statsbomb.com/what-we-do/hub/free-data/ <br>
[3] - https://github.com/metrica-sports/sample-data <br>


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


For more details check [ShotData/Scaling_Reasoning.xlsx](Scaling_Reasoning.xlsx). <br>