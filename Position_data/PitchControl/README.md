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


## class player

The player class is meant to bundle all steps that have to be executed for each player. <br>
A class object can be initiated manually but initiation via *get_player_from_data* or *get_all_players* is recommended!

**Attributes** 

+ 
+ *id (int)* - Player ID (as in the column names of td_object)
+ 


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
[7] - Fernández De La Rosa, J., Bornn, L., & Gavaldà Mestre, R. (2022). A framework for the analytical and visual interpretation of complex spatiotemporal dynamics in soccer [Universitat Politècnica de Catalunya]. https://doi.org/10.5821/dissertation-2117-363073
[8] - Fernandez, J., & Bornn, L. (2018). Wide Open Spaces: A statistical technique for measuring space creation in professional soccer. MIT Sloan Sports Analytics Conference. <br>
[9] - Fernández, J., Bornn, L., & Cervone, D. (2019). Decomposing the Immeasurable Sport: A deep learning expected possession value framework for soccer. MIT Sloan Sports Analytics Conference, Boston.
[10] - Fernández, J., Bornn, L., & Cervone, D. (2021). A framework for the fine-grained evaluation of the instantaneous expected value of soccer possessions. Machine Learning, 110(6), 1389–1427. https://doi.org/10.1007/s10994-021-05989-6
[11] - Alguacil, F. P., Fernandez, J., Arce, P. P., & Sumpter, D. (2020). Seeing in to the future: Using self-propelled particle models to aid player decision-making in soccer. MIT Sloan Sports Analytics Conference, Boston. <br>
[12] - Brefeld, U., Lasek, J., & Mair, S. (2019). Probabilistic movement models and zones of control. Machine Learning, 108(1), 127–147. https://doi.org/10.1007/s10994-018-5725-1 <br>
[13] - Llana, S., Madrero, P., Fernández, J., & Barcelona, F. (2020). The right place at the right time: Advanced off-ball metrics for exploiting an opponent’s spatial weaknesses in soccer. MIT Sloan Sports Analytics Conference, Boston. <br>
