# Shot Data

This folder contains a class (within *ShotData.py*) that helps to handle shot data (so far mainly Understat-data [1]
and Statsbomb [2]) from a single match which entails functions to create shotmaps and xG-flowcharts.
Those visualizations are inspired by the content from Understat [1]. For some examples check out the notebooks in this
directory.<br>
For insights on the structure of Statsbomb data and how to access either check 
https://github.com/DavidB1999/My_Football_Analytics/tree/main/Basics/Data or https://github.com/statsbomb/statsbombpy! <br>

## **class shot_data**

**Attributes**

+ *data (dataframe)*: dataframe containing the columns with the team name, minute, xG-values and x and y coordinate 
+ *data_source (str)* - data origin (influences data handling)
+ *x_range_data* and *y_range_data (tuple, numeric)* - range of x and y coordinates in the *shot_data*
+ *team_column (str)* - name of the column containing team names
+ *x_col (str)* - name of the column containing the x-coordinates
+ *y_col (str)* - name of the column containing the y-coordinates
+ *xg_column (str)* - name of the column containing the xG-values
+ *minute_col (str)* - name of the column containing minutes
+ *result_col (str)* - name of the column containing the outcome of each shot
+ *player_col (str)* - name of the column containing the name of the player taking the shot
+ *scale_to_pitch (str)* - default options of data ranges to scale to
+ *x_range_pitch* and *x_range_pitch (tuple, numeric)* - range of x and y coordinates required
+ *mirror_away (list, str)* - list of axes (x, y) to mirror for the away team 
+ *location_column (str)* - name of the column containing both coordinates (if the coordinates are in a single column - Statsbomb)
+ *shot_column (str)* - name of the column containing a dictionary with information on the shot (Statsbomb)
+ *xg_key (str)* - key for xG-value in the shot information dictionary (shot_column; Statsbomb)
+ *outcome_key (str)* - key for outcome in the shot information dictionary (shot_column; Statsbomb)
+ *teams (list, str)* - optional list to indentify home and away team; if None the code will assume that team first in the list of shots is the home team (works for Understat) 


### **rescale_shot_data**
                      (self) 
Function to handle shot data. The main purpose is the rescaling of x and y coordinates to match to a certain
data range as for instance necessary for pitches. An indepth explanation of the rescaling logic used can be found
in the Appendix. <br>

**Parameters** 

**Returns** 

+ data (dataframe) - rescaled shot_data



### **count_goals**
                (self, team)
A simple utility function to count the number of goals for a team based on the outcome of all shots. <br>

**Parameters** 

+ *team (str)* - either 'home' or 'away' - indicating the team for which the number of goals is to be counted

**Returns**

+ *ngoals (int)* - the number of goals scored


### **xG_score**
                (self, team)
A simple utility function to calculate the number of expected goals for a team as the sum of the xG of all shots. <br>

**Parameters** 

+ *team (str)* - either 'home' or 'away' - indicating the team for which the number of goals is to be counted

**Returns**

+ *xG_score (float)* - the number of expected goals


### **static_shotmap**
                    (self, pitch_type='mplsoccer', point_size_range=(20, 500),
                    markers={'SavedShot': "^", 'MissedShots': 'o', 'BlockedShot': "v", 'Goal': '*',
                                'OwnGoal': 'X', 'ShotOnPost': "h"},
                    alpha=0.5, color1='red', color2='blue',
                    xg_text=True, xg_text_x=None, xg_text_y=None,
                    result_text=True, result_text_x=None, result_text_y=None,
                    name_text=True, name_text_x=None, name_text_y=None,
                    home_image=None, away_image=None, logo_x=None, logo_y=None)  

**Parameters** 


+ *pitch_type (str)* - type of pitch used for the shotmap; currently supported are 'myPitch' (custom) and 'mplsoccer'
+ *point_size_range (tuple, float)* - lower and upper limit of scatter-point size 
+ *markers (dict)* - Dictionary assigning a marker-type to each outcome
+ *alpha (float[0,1])* - opacity of scatter-points
+ *color1* and *color2* - Colors for the two teams
+ *xg_text*, *result_text* and *name_text (boolean)* - Boolean indicating whether xG-score, final score and team names are supposed to be displayed
+ *xg_text_x*, *xg_text_y*, *result_text_x*, *result_text_y*, *name_text_x* and *name_text_y (float)* - x and y coordinates as fraction of the pitch limits; if *None* they will be placed based on pitch_type selection
+ *home_image* and *away_image (str)* - path to the teams' logos if to be displayed; if *None*, no logos will be displayed
+ *logo_x* and *logo_y (float)* - logos' x and y coordinates as fraction of the pitch limits; if *None* they will be placed based on pitch_type selectio

**Returns**

+*fig (figure)* - the shotmap

### interactive_shotmap
                        (self, color1='red', color2='blue', pitch_type='mplsoccer', background_col='#435348',
                        pitch_x0=None, pitch_y0=None, size_multiplicator=5, title=None, title_col='white',
                        xg_text=True, xg_text_x=None, xg_text_y=None, margins=None,
                        result_text=True, result_text_x=None, result_text_y=None,
                        name_text=True, name_text_x=None, name_text_y=None,
                        home_image=None, away_image=None, logo_x=None, logo_y=None,
                        axis_visible=False, pitch_path='')

**Parameters** 

+ *color1* and *color2* - Colors for the two teams
+ *pitch_type (str)* - type of pitch used for the shotmap; currently supported are 'myPitch' (custom) and 'mplsoccer'
+ *background_col* - color for the background around the shotmap
+ *pitch_x0* and *pitch_y0 (float)* - offset to the pitch (image) location on the plot to ensure the pitch to align with coordinates
+ *size_multiplicator (float)* - Used to multiply pitch size to increase figure size while maintaining aspect ratio
+ *title (str)* - Title to be displayed at the top
+ *title_col* - Color of the title
+ *xg_text*, *result_text* and *name_text (boolean)* - Boolean indicating whether xG-score, final score and team names are supposed to be displayed
+ *xg_text_x*, *xg_text_y*, *result_text_x*, *result_text_y*, *name_text_x* and *name_text_y (float)* - x and y coordinates as fraction of the pitch limits; if *None* they will be placed based on pitch_type selection
+ *home_image* and *away_image (str)* - path to the teams' logos if to be displayed; if *None*, no logos will be displayed
+ *logo_x* and *logo_y (float)* - logos' x and y coordinates as fraction of the pitch limits; if *None* they will be placed based on pitch_type selectio
+ *axis_visible (boolean)* - Boolean indicating whether axis are to be displayed
+ *pitch_path (str)* - Optional path (folder) to be added to the default saving path ('pitch.png'); Folder needs to exist and optional path needs to be suffixed by '/'

**Returns**

+ *fig (figure)* - the shotmap

### xg_chart
            (self, color1='red', color2='blue', Title=None, text_col='white', font_type='Rockwell',
            grid_visible=True, grid_col='#a3a3a3', plot_col='#999999', ball_image_path='Football3.png',
            display_score=True, home_image=None, away_image=None, design=None, ball_size_x=1.75, ball_size_y=0.1)

**Parameters** 

+ *color1* and *color2* - Colors for the two teams
+ *title (str)* - Title to be displayed at the top
+ *text_col* - Color for all text elements
+ *font_type* - Font family for all text elements
+ *grid_visible (boolean)* - Boolean indicating whether grid lines are to be seen
+ *grid_col* - Color of the grid
+ *plot_col* - Background color
+ *ball_image_path (str)* - path to the image being displayed for scored goals
+ *display_score (boolean)* - Boolean indicating whether final score is to be displayed
+ *home_image* and *away_image (str)* - path to the teams' logos if to be displayed; if *None*, no logos will be displayed
* *design (str)* - Optional selection of a pre-designed optic; overwrites varying layout parameters  
* *ball_size_x* and *ball_size_y* - Height and width of the image displayed for goals


**Returns**

+ *fig (figure)* - the xG flowchart


## References
[1] - https://understat.com/ <br>
[2] - https://github.com/statsbomb/statsbombpy | https://statsbomb.com/


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


For more details check [Scaling_Reasoning.xlsx](Scaling_Reasoning.xlsx). <br>