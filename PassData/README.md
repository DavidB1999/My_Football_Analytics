# Pass Data

This folder contains a class (within *Pass_Data.py*) that helps to handle pass data (so far mainly Statsbomb [2]) from
a single match which entails a function to filter and count the passes. <br>
For insights on the structure of Statsbomb data and how to access either check 
https://github.com/DavidB1999/My_Football_Analytics/tree/main/Basics/Data or https://github.com/statsbomb/statsbombpy! <br>

## **class pass_data**

**Attributes** <br>
If a data source is specified all attributes are defined accordingly. If necessary (e.g. changes in the Statsbomb 
naming conventions) all attributes can be specified manually. <br>

+ *data (dataframe)*: dataframe (usually from Statsbomb) containing all relevant columns 
+ *data_source (str)* - data origin (influences data handling)
+ *x_range_data* and *y_range_data (tuple, numeric)* - range of x and y coordinates in the *pass_data*
+ *team_column (str)* - name of the column containing team names
+ *location_col (str)*- name of the column containing x- and y- coordinates in tuples
+ *pass_col (str)* - name of the column containing a dictionary for each pass with additional information
+ *player_col (str)* - name of the column containing the name of the player making the pass
+ *teams (list, str)* - optional list to indentify home and away team; if None the code will assume that team first in the list of shots is the home team (works for Understat) 
+ *scale_to_pitch (str)* - default options of data ranges to scale to
+ *x_range_pitch* and *x_range_pitch (tuple, numeric)* - range of x and y coordinates required
+ *rel_eve_col* - name of the column containing the id of related events (like shots for xG / xA)
+ *mirror_away (list, str)* - list of axes (x, y) to mirror for the away team 
+ *type_key (str)* - key for pass type in the pass information dictionary (pass_column; Statsbomb)
+ *outcome_key (str)* - key for pass outcome in the pass information dictionary (pass_column; Statsbomb)
+ *minute_col (str)* - name of the column containing minutes
+ *second_col (str)* - name of the column containing second
+ *shot_ass_key (str)* - key for shot_assist outcome in the pass information dictionary (pass_column; Statsbomb)
+ *goal_ass_key (str)* - key for goal_assist outcome in the pass information dictionary (pass_column; Statsbomb)
+ *cross_key (str)* - key for crosses in the pass information dictionary (pass_column; Statsbomb)
+ *cutback_key (str)* - key for cut-backs in the pass information dictionary (pass_column; Statsbomb)
+ *switch_key (str)* - key for switches in the pass information dictionary (pass_column; Statsbomb)
+ *play_pattern_col* - name of the column containing the information on play pattern
+ *half_col* - name of the column indicating the half/period of player


### **rescale_pass_data**
                      (self) 
Function to handle pass data. The main purpose is the rescaling of x and y coordinates to match to a certain
data range as for instance necessary for pitches. An indepth explanation of the rescaling logic used can be found
in the Appendix. 'Unnecessary' columns are not retained in rescaled data. <br>

**Parameters** 

**Returns** 

+ data (dataframe) - rescaled pass_data


### **get_passes**
                (self, get, data=None, receiver_get=False, receiver_count=False)

Allows to filter the passes by most informational elements. <br>

**Parameters**

+ *get (str, int)* - Any string (or int for 'period') indicating a filter condition.
+ *data (dataframe)* - Dataframe to be filtered. If None the class' data will be used. Specifying a dataframe allows to apply multiple filters in sequence
+ *receiver_get (boolean)* - Is the name supplied to get to be used on the receivers instead of the passing players? Meaning: Should filter be applied to receiving players?
+ *receiver_count (boolean)* - Is the count that is returned supposed to distinguish between receiving players instead of passing players? 

### **pass_map**
                (self, plot_direction_of_play=True, data=None, direction_of_play='ltr', pdop_x=1 / 3, pdop_y=0.1,
                 pdop_l=1 / 3, pitch_col='#1c380e', line_col='white', pdop_o=0.2)

**Parameters** 

+ *plot_direction_of_play (boolean)* - Indicating if possession of play should be displayed by an arrow (only possible if just one team is included)
+ *data (dataframe)* - Dataframe with passes to be plotted. If None the class' data will be used. Specifying a dataframe allows to apply multiple filters and is recommended!
+ *direction_of_play (str)* - "ltr" for left to right or "rtl" for right to left, indicating the correct direction of play
+ *pdop_x* and *pdop_y (float)* - numerical indicating the relative placement of direction of play arrow
+ *pdop_y (float)* - numerical indicating the relative length of the direction of play arrow
+ *pdop_o (float)* - numerical indicating the opacity/alpha of the direction of play arrow
+ *pitch_col (color)* - color of the pitch plotted
+ *line_col (color)* - color of the pitch lines

**Returns**

+ *fig (figure)* - pass map figure

### **pass_network**
                    (self, pitch_col='#1c380e', line_col='white', colors=None, data=None, pass_min=5,
                     by_receive=False)

**Parameters** 

+ *pitch_col (color)* - color of the pitch plotted
+ *line_col (color)* - color of the pitch lines
+ *colors* - list of colors for the 11 players
+ *data (dataframe)* - Dataframe with passes to be plotted. If None the class' data will be used. Specifying a dataframe allows to apply multiple filters and is recommended!
+ *pass_min* - Minimum of passes played for a connection to be displayed in the network
+ *by_receive* - If True average positions will be calculated based on the average receiving position instead of pass origins

**Returns**

+ *network (dict)* - information on connections
+ *fig (figure)* - pass network figure


### **count_returner**
                     (self, data, receiver=False)

Function to return number of passes per unit.
Unit is automatically determined based on the number of players, and teams in the supplied data. <br>

**Parameters** 

+ *data (dataframe)* - Dataframe with passes to be plotted. 
+ *receiver (boolean)* - Is the count that is returned supposed to distinguish between receiving players instead of passing players? (receiver_count)

**Returns**

+ *n (int or dict)* - count per unit



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




