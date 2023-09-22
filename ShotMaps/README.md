# Shot maps

This folder contains everything necessary to create shot maps inspired by the shotmaps from understat [1]. <br>

## ShotMaps.py

### **rescale_shot_data**
                      (shot_data, data_source=None, x_range_data=None, y_range_data=None, team_column='team',
                      x_col='x', y_col='y', xg_col='xG', minute_col='minute', scale_to_pitch='mplsoccer',
                      x_range_pitch=None, y_range_pitch=None, mirror_away=['x', 'y']) 

**Parameters** 

+ *shot_data (dataframe)*: dataframe containing the columns with the team name, minute, xG-values and x and y coordinate 
+ *data_source (str)* - data origin (influences data handling)
+ *x_range_data* and *y_range_data (tuple, numeric)* - range of x and y coordinates in the *shot_data*
+ *team_column (str)* - name of the column containing team names
+ *x_col (str)* - name of the column containing the x-coordinates
+ *y_col (str)* - name of the column containing the y-coordinates
+ *xg_column (str)* - name of the column containing the xG-values
+ *minute_col (str)* - name of the column containing minutes
+ *scale_to_pitch (str)* - default options of data ranges to scale to
+ *x_range_pitch* and *x_range_pitch (tuple, numeric)* - range of x and y coordinates required
+ *mirror_away (list, str)* - list of axes (x, y) to mirror for the away team 

**Return** 

+ data (dataframe) - rescaled shot_data


**Rescaling logic:** <br>

$$ shotdataminmax = (a_1, a_2)$$

$$ pitchminmax = (b_1, b_2)$$

$$ \Delta a = a_2- a_1$$ 

$$ \Delta b = b_2- b_1$$ 

$$ c = coordinate$$

$$ scalingfactor = s = \frac{\Delta b}{\Delta a}$$

**Without mirroring:** <br>

$$ \Delta a > 0: c_{new}  = b_1 + c \times s$$

$$ \Delta a < 0: c_{new}  = b_2 + c \times s$$

**With mirroring:** <br>

$$ \Delta a > 0: c_{new}  = b_1 - c \times s$$

$$ \Delta a < 0: c_{new}  = b_2 - c \times s$$

For more details check [Scaling_Reasoning.xlsx](Scaling_Reasoning.xlsx). <br>


### static_shotmap

to follow <br>

### plotly_shotmap

to follow <br>



## References
[1] - https://understat.com/ <br>
