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
                (self, get, data=None)

Allows to filter the passes by most informational elements. <br>

**Parameters**

+ *get (str, int)* - Any string (or int for 'period') indicating a filter condition.
+ data (dataframe) - Dataframe to be filtered. If None the class' data will be used. Specifying a dataframe allows to apply multiple filters in sequence

