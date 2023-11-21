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
An alternative way to implement Spearman's pitch control based on the code from "anenglishgoat" uses tensors (pytorch)
and offers the option to use "gpu"-support (if available) to speed up the computation process. <br>
 
### get_player_from_data
                        (td_object, pid, team, data=None, frame=None, params=None)

This function has no direct equivalent in Shaw's code. 
It simply returns a player class object for a player selected by ID from a tracking data class object. <br>

**Parameters**

+ *td_object (tracking_data class object)* - An object of the tracking_data class containing data and all required attributes 
+ *pid (int)* - player ID (as in the column names of td_object)
+ *team (str)* - the player's team (either "Home" or "Away")
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
+ *kappa_gk (float)* - "make goalkeepers must quicker to control ball (because they can catch it)"
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


### animate_pitch_control
                        (td_object, start_frame, end_frame, attacking_team='Home', velocities=False, params=None,
                         n_grid_cells_x=50, frames_per_second=25, fname='Animated_Clip', pitch_col='#1c380e',
                         line_col='white', colors=['red', 'blue', 'black'], PlayerAlpha=0.7, fpath=None,
                         progress_steps = [0.25, 0.5, 0.75])

Function to animate a set of frames of plotted players including pitch control. <br>
This can take quite a while so a small number of frames and the use of *progress_steps* is recommended! <br>

**Parameters**

+ *td_object (tracking_data class object)* - An object of the tracking_data class containing data and all required attributes 
+ *start_frame*, *end_frame (int)* - Frame range for the animation
+ *attacking_team (str)* - which team is to be interpreted as attacking team (optimally in possession at frame); home or away
+ *velocites (boolean)* - decides whether players' velocities will be plotted
+ *params (dict)* - Model params dictionary as created by function *default_model_params*; if None default parameters will be used in *pitch_control_at_frame*
+ *n_grid_cells_x (int)* - number of cells in the grid on the x-axis; number of grid cells in y direction will be determined based on n_grid_cells_x and pitch dimensions
+ *frames_per_second (int)* - frames per second to assume when generating the movie. Default is 25.
+ *fname (str)* - intended file name to store the clip. Defaults to "Animated_Clip"
+ *pitch_col (color)* - color of the pitch
+ *line_col (color)* - color of the pitch lines
+ *colors (list, color)* - list of colors for home team, away team and the ball (in that order)
+ *PlayerAlpha (float)* - Opacity/alpha of velocity quiver
+ *fpath (str)* - Directory to save the Clip. If None Clip will be stored in current directory.
+ *progress_steps (list, float)* - percentages (in decimals) to be displayed as progress steps; if None no progress will be displayed

**Returns**

### tensor_pitch_control
                        (td_object, jitter=1e-12, pos_nan_to=-1000, vel_nan_to=0, remove_first_frames=0,
                        reaction_time=0.7, =5, average_ball_speed=15, sigma=0.45, lamb=4.3,
                        n_grid_points_x=50, n_grid_points_y=30, device='cpu', dtype=torch.float32,
                        first_frame=0, last_frame=500, batch_size=250, deg=50, version='GL', max_int=500,
                        team='Home')


Function calculate the pitch control of the home team in a given range of frames. Function relies on tracking_data class
object from my tracking data package. Function is based on the code by "anenglishgoat".

**Parameters**

+ *td_object (tracking_data class object)* - An object of the tracking_data class containing data and all required attributes 
+ *jitter (float)* - minimal value added to players velocity to avoid devision by zero
+ *pos_nan_to (float)* - Value to replace missing values in position data with. Default is -1000 to render the impact of inactive players to basically 0.
+ *vel_nan_to (float)* - Value to replace missing values in velocity data with. Default is 0 to render the impact of inactive players to basically 0.
+ *remove_first_frames (float)* - Allows to skip first n frames of selected frame range; fromDefault is 0; superfluous when using *first_frame* and *last_frame*
+ *reaction_time (float)* - "seconds, time taken for player to react and change trajectory. Roughly determined as vmax/amax"
+ *average_ball_speed (float)* - "average ball travel speed in m/s"
+ *max_player_speed (float)* - "maximum player speed m/s"
+ *sigma (float)* - time to interception sigma: "Standard deviation of sigmoid function in Spearman 2018 ('s') that determines uncertainty in player arrival time" - = Spearman's sigma != anenglishgoat's sigma (see exp)
+ *lamb (float)* - "ball control parameter"
+ *n_grid_points_x*, *n_grid_points_y (int)* - number of pitch control target locations in both dimensions
+ *device (str)* - device used for computation; if available "gpu" can speed up the process; for more information see https://pytorch.org/docs/stable/generated/torch.cuda.device.html
+ *dtype (str)* - datatype used in torch tensors; for more information see https://pytorch.org/docs/stable/tensor_attributes.html
+ *first_frame*, *last_frame (int)* - frame interval over which the pitch control is supposed to be modelled
+ *batch_size (int)* - batch size used for tensors and computational process; instead of looping over frames we loop over batches containing batch_size number of frames
+ *deg (int)* - Number of sample points and weights for numpy.polynomial.legendre.leggauss (https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.leggauss.html)
+ *version (str)* - Computation version. So far only the Gauss-Legendre quadrature ('GL') version is included. An classical integration version should follow.
+ *max_int (int)* - maximal interval length for integration method
+ *team (str)* - "Team-perspective" for pitch control modeling - Either "Home" or "Away"

**Returns**

+ *pc (torch.tensor)* - Pitch control for home team of shape (n_frames, *n_grid_points_x*, *n_grid_points_y* ) covering every target location on the pitch in all of the selceted frames


### plot_tensor_pitch_control
                            (td_object, frame, pitch_control, jitter=1e-12, pos_nan_to=-1000, vel_nan_to=0,
                            remove_first_frames=0, reaction_time=0.7, max_player_speed=5, average_ball_speed=15,
                            sigma=0.45, lamb=4.3, n_grid_points_x=50, n_grid_points_y=30, device='cpu',
                            dtype=torch.float32, first_frame=0, last_frame=500, batch_size=250, deg=50, version='GL',
                            cmap='bwr', velocities=True, max_int=500, team='Home')

Function to plot players and pitch control a pitch. Uses the *plot_players* from the tracking data class and the 
*tensor_pitch_control* function. 

**Parameters**

+ *td_object (tracking_data class object)* - An object of the tracking_data class containing data and all required attributes 
+ *frame (int)* - frame to be plotted (absolute frame number)
+ *pitch_control (torch.tensor)* - Output from *tensor_pitch_control* function. *tensor_pitch_control* will only be called if None 
+ *jitter (float)* - minimal value added to players velocity to avoid devision by zero
+ *pos_nan_to (float)* - Value to replace missing values in position data with. Default is -1000 to render the impact of inactive players to basically 0.
+ *vel_nan_to (float)* - Value to replace missing values in velocity data with. Default is 0 to render the impact of inactive players to basically 0.
+ *remove_first_frames (float)* - Allows to skip first n frames of selected frame range; fromDefault is 0; superfluous when using *first_frame* and *last_frame*
+ *reaction_time (float)* - "seconds, time taken for player to react and change trajectory. Roughly determined as vmax/amax"
+ *average_ball_speed (float)* - "average ball travel speed in m/s"
+ *max_player_speed (float)* - "maximum player speed m/s"
+ *sigma (float)* - time to interception sigma: "Standard deviation of sigmoid function in Spearman 2018 ('s') that determines uncertainty in player arrival time" - = Spearman's sigma != anenglishgoat's sigma (see exp)
+ *lamb (float)* - "ball control parameter"
+ *n_grid_points_x*, *n_grid_points_y (int)* - number of pitch control target locations in both dimensions
+ *device (str)* - device used for computation; if available "gpu" can speed up the process; for more information see https://pytorch.org/docs/stable/generated/torch.cuda.device.html
+ *dtype (str)* - datatype used in torch tensors; for more information see https://pytorch.org/docs/stable/tensor_attributes.html
+ *first_frame*, *last_frame (int)* - frame interval over which the pitch control is supposed to be modelled
+ *batch_size (int)* - batch size used for tensors and computational process; instead of looping over frames we loop over batches containing batch_size number of frames
+ *deg (int)* - Number of sample points and weights for numpy.polynomial.legendre.leggauss (https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.leggauss.html)
+ *version (str)* - Computation version. Allows for both the Gauss-Legendre quadrature ('GL') version and an classical integration version ('int').
+ *cmap (str)* - color map used for the pitch control visualization
+ *velocities (boolean)* - Whether velocities are supposed to be displayed
+ *flip_y (boolean)* - Indicates whether the pitch control grid needs to be flipped on y-axis. 
+ *max_int (int)* - maximal interval length for integration method
+ *team (str)* - "Team-perspective" for pitch control modeling - Either "Home" or "Away"
+ 
**Returns**

+ *fig (figure)* - player positions and pitch control displayed on a pitch

### pos_to_array
                (pos_data, nan_to, ball=False)

Function to convert data frame to array as required in tensor_pitch_control.

**Parameters** 

+ *pos_data (pd.dataframe)* - Input Dataframe  
+ *nan_to (float)* - Value to replace missing values in data with.
+ *ball (boolean)* - Boolean indicating whether the input data is ball position data (i.e. 2 columns)

**Returns**

+ *array (np.ndarray)* - Input data as an array


### animate_tensor_pitch_control
                                (td_object, pitch_control=None, jitter=1e-12, pos_nan_to=-1000, vel_nan_to=0,
                                 remove_first_frames=0, reaction_time=0.7, max_player_speed=5, average_ball_speed=15,
                                 sigma=0.45, lamb=4.3, n_grid_points_x=50, n_grid_points_y=30, device='cpu',
                                 dtype=torch.float32, first_frame_calc=0, last_frame_calc=500, batch_size=250, deg=50,
                                 version='GL', cmap='bwr', velocities=True, flip_y=True, 
                                 progress_steps=[0.25, 0.5, 0.75], frames_per_second=None, fpath=None,
                                 fname='Animation', pitch_col='#1c380e', line_col='white',
                                 colors=['red', 'blue', 'black'], PlayerAlpha=0.7, first_frame_ani=0,
                                 last_frame_ani=100, max_int=500, team='Home')

Function to create an animation of player and ball position and pitch control over a given range of frames.

**Parameters**

+ *td_object (tracking_data class object)* - An object of the tracking_data class containing data and all required attributes 
+ *pitch_control (torch.tensor)* - Output from *tensor_pitch_control* function. *tensor_pitch_control* will only be called if None 
+ *jitter (float)* - minimal value added to players velocity to avoid devision by zero
+ *pos_nan_to (float)* - Value to replace missing values in position data with. Default is -1000 to render the impact of inactive players to basically 0.
+ *vel_nan_to (float)* - Value to replace missing values in velocity data with. Default is 0 to render the impact of inactive players to basically 0.
+ *remove_first_frames (float)* - Allows to skip first n frames of selected frame range; fromDefault is 0; superfluous when using *first_frame* and *last_frame*
+ *reaction_time (float)* - "seconds, time taken for player to react and change trajectory. Roughly determined as vmax/amax"
+ *average_ball_speed (float)* - "average ball travel speed in m/s"
+ *max_player_speed (float)* - "maximum player speed m/s"
+ *sigma (float)* - time to interception sigma: "Standard deviation of sigmoid function in Spearman 2018 ('s') that determines uncertainty in player arrival time" - = Spearman's sigma != anenglishgoat's sigma (see exp)
+ *lamb (float)* - "ball control parameter"
+ *n_grid_points_x*, *n_grid_points_y (int)* - number of pitch control target locations in both dimensions
+ *device (str)* - device used for computation; if available "gpu" can speed up the process; for more information see https://pytorch.org/docs/stable/generated/torch.cuda.device.html
+ *dtype (str)* - datatype used in torch tensors; for more information see https://pytorch.org/docs/stable/tensor_attributes.html
+ *first_frame_calc*, *last_frame_calc (int)* - frame interval over which the pitch control is supposed to be modelled (interval should include animation interval!)
+ *batch_size (int)* - batch size used for tensors and computational process; instead of looping over frames we loop over batches containing batch_size number of frames
+ *deg (int)* - Number of sample points and weights for numpy.polynomial.legendre.leggauss (https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.leggauss.html)
+ *version (str)* - Computation version. Allows for both the Gauss-Legendre quadrature ('GL') version and an classical integration version ('int').
+ *cmap (str)* - color map used for the pitch control visualization
+ *velocities (boolean)* - Whether velocities are supposed to be displayed
+ *flip_y (boolean)* - Indicates whether the pitch control grid needs to be flipped on y-axis. 
+ *progress_steps (list, float)* - percentages (in decimals) to be displayed as progress steps; if None no progress will be displayed
+ *frames_per_second (int)* - frames per second to assume when generating the movie. Default is 25.
+ *fpath (str)* - Directory to save the Clip. If None Clip will be stored in current directory.
+ *fname (str)* - intended file name to store the clip. Defaults to "Animated_Clip"
+ *pitch_col (color)* - color of the pitch
+ *line_col (color)* - color of the pitch lines
+ *colors (list, color)* - list of colors for home team, away team and the ball (in that order)
+ *PlayerAlpha (float)* - Opacity/alpha of velocity quiver
+ *first_frame_ani*, *last_frame_ani (int)* - frame interval to be animated(interval should be within calculation interval!)
+ *max_int (int)* - maximal interval length for integration method
+ *team (str)* - "Team-perspective" for pitch control modeling - Either "Home" or "Away"



## Credits

Laurie Shaw's excellent GitHub at friends of tracking - https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking/tree/master -
and the video series - https://www.youtube.com/watch?v=VX3T-4lB2o0 - <br>
Data Metrica - https://github.com/metrica-sports/sample-data <br>
Code by "anenglishgoat" - https://github.com/anenglishgoat/Metrica-pitch-control <br>


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
