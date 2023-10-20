# Pitch control

This is supposed to be a deep dive into pitch control and related concepts. - long term project <br>
At this stage only a version Spearman's pitch control model is implemented within this project. <br>

There are many ways to implement pitch control / space control. Some of the most prominent are:

+ Voronoi -  https://www.youtube.com/watch?v=MIGQFcVO7-s, [1, 2, 3]
+ Spearman - https://www.youtube.com/watch?v=X9PrwPyolyU  [4, 5, 6]
+ Fernandez - [7, 8 , 9, 10]

Other relevant publications include the following: [11, 12, 13]. <br>
Some interesting inside into the implementation of both Spearman's and Fernandez' modelling approach can be found online
in video material and Code on GitHub: https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking | 
https://github.com/anenglishgoat/Metrica-pitch-control | https://www.youtube.com/watch?v=X9PrwPyolyU. <br>
The related concept of EPV is also covered online by Laurie Shaw: https://www.youtube.com/watch?v=KXSLKwADXKI |  
https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking<br>


# pitch_control.py

This scripts contains all necessary steps and functions to model pitch control on the basis of an object of my tracking data class. 
The code is based entirely on Laurie Shaw's code and videos and was adapted to work my tracking data functionality. 
All adaptations were implemented for that purpose only and did not change anything of the underlying mechanism (unless I made a mistake :worried:)!<br>

### get_player_from_data
                        (td_object, pid, data=None, frame=None, params=None)

This function has no direct equivalent in Shaw's code. 
It simply returns a player class object for a player selected by ID from a tracking data class object. <br>

**Parameters**

+ *td_object (tracking_data class object)* - An object of the tracking_data class containing data and all required attributes 
+ *pid (int)* - player ID (as in the column names of td_object)
+ *data (dataframe)* - To specify a dataframe; if None, the data of td_object will be used 
+ *frame (int)* - To select a frame from data; if None, the entire data for the player will be used
+ *params (dict)* - A dictionary with model parameters; if None default model parameters will be created by intended function and used

**Returns**

+ *p_object (player class object)* - An object of the player class object containing the data and all required attributes



### get_all_players
                def get_all_players(td_object, frame=None, teams=['Home', 'Away'], params=None)

Functional equivalent to Shaw's *initialise_players*-function, creating a player class object for all players. <br>

**Parameters**

+ *td_object (tracking_data class object)* - An object of the tracking_data class containing data and all required attributes
+ *frame (int)* - To select a frame from data; if None, the entire data for the player will be used
+ *teams (list, str)* - List with the teams of which we want the players
+ *params (dict)* - A dictionary with model parameters; if None default model parameters will be created by intended function and used

**Returns**

+ *players (list)* - List of objects of the player class object containing the data and all required attributes




### default_model_params
                        (time_to_control_veto=3, mpa=7, mps=5, rt=0.7, tti_s=0.45, kappa_def=1,       
                         lambda_att=4.3, kappa_gk=3, abs=15, dt=0.04, mit=10, model_converge_tol=0.01)

Function to define standard parameters for the pitch control modelling process. Largely the same as Shaw's function. <br>
Parameter descriptions are copied from https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking/blob/master/Metrica_PitchControl.py. 

**Parameters**

+ *time_to_control_veto (float)* - "If the probability that another team or player can get to the ball and control it is less than 10^-time_to_control_veto, ignore that player."
+ *mpa (float)* -  "maximum player acceleration m/s/s, not used in this"
+ *mps (float)* -  "maximum player speed m/s"
+ *rt (float)* - reaction time: "seconds, time taken for player to react and change trajectory. Roughly determined as vmax/amax"
+ *tti_s (float)* - time to interception sigma: "Standard deviation of sigmoid function in Spearman 2018 ('s') that determines uncertainty in player arrival time"
+ *kappa_def (float)* - "kappa parameter in Spearman 2018 (=1.72 in the paper) that gives the advantage defending players to control ball, I have set to 1 so that home & away players have same ball control probability"
+ *lambda_att (float)* - "ball control parameter for attacking team"
+ *kappa_gk (float)* - "make goal keepers must quicker to control ball (because they can catch it)"
+ *abs (float)* - average ball speed: "average ball travel speed in m/s"
+ *dt (float)* - int_dt: "integration timestep (dt)"
+ *mit (float)* - max_int_time: "upper limit on integral time"
+ *model_converge_tol (float)* - "assume convergence when PPCF>0.99 at a given location"

**Returns**

+ *params (dict)* - dictionary of model parameters

----------------------------------------------------------------------------------------------------------------------

## class player

The player class is meant to bundle all steps that have to be executed for each player. <br>
A class object can be initiated manually but initiation via *get_player_from_data* or *get_all_players* is recommended!

**Attributes** 

for (manual) initiation:

+ *data (dataframe)* - player's tracking data
+ *pid (int)* - Player ID (as in the column names of td_object)
+ *GK (boolean)* - Indicating whether player is a goalkeeper
+ *team (str)* - Is the player from the home or away team?
+ *params (dict)* - Model params dictionary as created by function *defaul_model_params*
+ *frame (int)* - Frame number if just one frame

other attributes:

+ *id (int)* - Player ID (as in the column names of td_object)
+ *org_data (dataframe)* - tracking data as supplied in initiation
+ *team (str)* - Is the player from the home or away team?
+ *GK (boolean)* - Indicating whether player is a goalkeeper
+ *frame (int)* - Frame number if just one frame
+ *params (dict)* - Model params dictionary as created by function *defaul_model_params*; if None default params will be used
+ *player_name (str)* - Not the actual name! Combination of team and id to identify relevant columns
+ *position (dataframe or list, float)* - position of player
+ *inframe (boolean)* - is the player in the frame in question or missing values?
+ *velocity (dataframe or list, float)* - velocity of player in both x and y direction
+ *time_to_intercept (float)* - time to intercept estimation
+ *PPCF (float)*  - player's contribution to pitch control at target location


### get_position
                (self)

**Parameters**

**Returns**

+ *position (dataframe or list, float)* - position of player
+ *inframe (boolean)* - is the player in the frame in question or missing values?


### get_velocity
                (self)

**Parameters**

**Returns**

+ *velocity (dataframe or list, float)* - velocity of player in both x and y direction


### simple_time_to_intercept
                            (self, r_final)
Calculates expected time to intercept based on equation of motion as in Spearman (2017) Equation 2. <br>

**Parameters**

+ *r_final (np.array, float)* - target position, i.e. how fast can the player get to r_final

**Returns**

+ + *time_to_intercept (float)* - time to intercept estimation


### probability_intercept_ball 
                            (self, T, include_time_to_intercept_calc=False, r_final=None)
Calculates the probability that the player will be able to intercept the ball based on a given time T. Equation 4 in Spearman (2018). <br>

**Parameters** 

+ *T (float)* - time it takes the ball to arrive at target location
+ *include_time_to_intercept_calc (boolean)* - whether *simple_time_to_intercept* is to be called based on an r_final or is already given. It is already given (and should be) when pitch control model is used. 
+ *r_final (np.array, float)* - target position, i.e. how fast can the player get to r_final

**Returns**

+ *f (float)* - Probability that the player can intercept the ball

----------------------------------------------------------------------------------------------------------------------

### pitch_control_at_frame
                        (frame, td_object, n_grid_cells_x=50, offside=False, attacking_team='Home', params=None)

Function to calculate pitch control for entire pitch at a specific frame. Identical to Laurie Shaw's method but adapted to fit my workflow.<br>
Calls *pitch_control_at_target* for each position of the grid. <br>

**Parameters**

+ *frame (int)* - To select a frame from data
+ *td_object (tracking_data class object)* - An object of the tracking_data class containing data and all required attributes
+ *n_grid_cells_x (int)* - number of cells in the grid on the x-axis; number of grid cells in y direction will be determined based on n_grid_cells_x and pitch dimensions
+ *offside (boolean)* - should attacking players that are offside be excluded from pitch control
+ *attacking_team (str)* - which team is to be interpreted as attacking team (optimally in possession at frame); home or away
+ *params (dict)* - Model params dictionary as created by function *default_model_params*; if None default parameters will be used

**Returns**

+ *PPCFa (numpy.ndarray, float)* - attacking team's pitch control grid
+ *xgrid, ygrid (numpy.ndarray, float)* - array of x and y values of the grid


### pitch_control_at_target
                        (target_position, attacking_players, defending_players, ball_start_pos, params=None)

Function to calculate the pitch control for both teams at a given target location. Identical to Laurie Shaw's method but adapted to fit my workflow. <br>

**Parameters**

+ *target_position (np.array, float)* - position for which we want the control modelled
+ *attacking players (list, player class objects)* - player class objects for all attacking players
+ *defending players (list, player class objects)* - player class objects for all defending players
+ *ball_start_pos (list, float)* - initial position of the ball
+ *params (dict)* - Model params dictionary as created by function *default_model_params*; if None default parameters will be used

**Returns**

+ *PPCFatt[i - 1], PPCFdef[i - 1] (floats)* - pitch control for both teams at target position 


### plot_pitch_control
                (td_object, frame, attacking_team='Home', PPCF=None, velocities=False, params=None, n_grid_cells_x=50)
Function to plot players and pitch control on a pitch. Calls *pitch_control_at_frame* and  *plot_players* from tracking data class object.<br>

**Parameters**

+ *td_object (tracking_data class object)* - An object of the tracking_data class containing data and all required attributes 
+ *frame (int)* - To select a frame from data
+ *attacking_team (str)* - which team is to be interpreted as attacking team (optimally in possession at frame); home or away
+ *PPCF (numpy.ndarray, float)* - A team's pitch control grid; if None it will be calculated via calling *pitch_control_at_frame*
+ *velocites (boolean)* - decides whether players' velocities will be plotted
+ *params (dict)* - Model params dictionary as created by function *default_model_params*; if None default parameters will be used in *pitch_control_at_frame*
+ *n_grid_cells_x (int)* - number of cells in the grid on the x-axis; number of grid cells in y direction will be determined based on n_grid_cells_x and pitch dimensions

**Returns**

+ *fig, ax (figure)* - pitch control plot



## Credits

Laurie Shaw's excellent GitHub at friends of tracking - https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking/tree/master -
and the video series - https://www.youtube.com/watch?v=VX3T-4lB2o0 - <br>



## References

[1] - Taki, T., & Hasegawa, J. (2000). Visualization of dominant region in team games and its application to teamwork analysis. Proceedings Computer Graphics International 2000, 227–235. https://doi.org/10.1109/CGI.2000.852338 <br>
[2] - Kim, S. (2004). Voronoi Analysis of a Soccer Game. Nonlinear Analysis: Modelling and Control, 9(3), 233–240. https://doi.org/10.15388/NA.2004.9.3.15154 <br>
[3] - Rein, R., Raabe, D., & Memmert, D. (2017). “Which pass is better?” Novel approaches to assess passing effectiveness in elite soccer. Human Movement Science, 55, 172–181. https://doi.org/10.1016/j.humov.2017.07.010 <br>
[4] - Spearman, W. (2016). Quantifying Pitch Control. OptaProForum. <br>
[5] - Spearman, W. (2018). Beyond Expected Goals. MIT Sloan Sports Analytics Conference. <br>
[6] - Spearman, W., Basye, A., Dick, G., Hotovy, R., & Pop, P. (2017). Physics-Based Modeling of Pass Probabilities in Soccer. MIT Sloan Sports Analytics Conference. <br>
[7] - Fernández De La Rosa, J., Bornn, L., & Gavaldà Mestre, R. (2022). A framework for the analytical and visual interpretation of complex spatiotemporal dynamics in soccer [Universitat Politècnica de Catalunya]. https://doi.org/10.5821/dissertation-2117-363073 <br>
[8] - Fernandez, J., & Bornn, L. (2018). Wide Open Spaces: A statistical technique for measuring space creation in professional soccer. MIT Sloan Sports Analytics Conference. <br>
[9] - Fernández, J., Bornn, L., & Cervone, D. (2019). Decomposing the Immeasurable Sport: A deep learning expected possession value framework for soccer. MIT Sloan Sports Analytics Conference, Boston. <br>
[10] - Fernández, J., Bornn, L., & Cervone, D. (2021). A framework for the fine-grained evaluation of the instantaneous expected value of soccer possessions. Machine Learning, 110(6), 1389–1427. https://doi.org/10.1007/s10994-021-05989-6 <br>
[11] - Alguacil, F. P., Fernandez, J., Arce, P. P., & Sumpter, D. (2020). Seeing in to the future: Using self-propelled particle models to aid player decision-making in soccer. MIT Sloan Sports Analytics Conference, Boston. <br>
[12] - Brefeld, U., Lasek, J., & Mair, S. (2019). Probabilistic movement models and zones of control. Machine Learning, 108(1), 127–147. https://doi.org/10.1007/s10994-018-5725-1 <br>
[13] - Llana, S., Madrero, P., Fernández, J., & Barcelona, F. (2020). The right place at the right time: Advanced off-ball metrics for exploiting an opponent’s spatial weaknesses in soccer. MIT Sloan Sports Analytics Conference, Boston. <br>
