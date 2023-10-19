import pandas as pd
import numpy as np
from Tracking_Data import tracking_data  # wrong here but necessary for use in notebooks
import matplotlib.pyplot as plt


# class player to bundle all steps for each player

def get_player_from_data(td_object, pid, data=None, frame=None, params=None):
    if params is None:
        params = default_model_params()
    if data is None:
        data = td_object.data
    GK = str(pid) in [td_object.Home_GK, td_object.Away_GK]
    if frame is None:
        player_data = data.filter(regex=fr'_{pid}_|Period|Time')
        team = player_data.columns[2].split('_')[0]
        p_object = player(data=player_data, pid=pid, GK=GK, team=team, params=params, frame=frame)
    else:
        player_data = data.filter(regex=fr'_{pid}_|Period|Time')
        player_data = player_data.loc[frame]
        team = player_data.keys()[2].split('_')[0]
        p_object = player(data=player_data, pid=pid, GK=GK, team=team, params=params, frame=frame)
    return p_object


def get_all_players(td_object, frame=None, teams=['Home', 'Away'], params=None):
    if params is None:
        params = default_model_params()
    data = td_object.data
    player_ids = np.unique([c.split('_')[1] for c in data.keys() if c[:4] in teams])
    players = []
    for p in player_ids:
        c_player = get_player_from_data(td_object=td_object, pid=p, frame=frame, params=params)
        if c_player.inframe:  # to check data is not nan i.e. on the pitch
            players.append(c_player)
    return players


def default_model_params(time_to_control_veto=3, mpa=7, mps=5, rt=0.7, tti_s=0.45, kappa_def=1,
                         lambda_att=4.3, kappa_gk=3, abs=15, dt=0.04, mit=10, model_converge_tol=0.01):
    params = dict()
    # model parameters
    params['max_player_accel'] = mpa  # maximum player acceleration m/s/s, not used in this
    # implementation
    params['max_player_speed'] = mps  # maximum player speed m/s 
    params['reaction_time'] = rt  # seconds, time taken for player to react and change
    # trajectory. Roughly determined as vmax/amax
    params['tti_sigma'] = tti_s  # Standard deviation of sigmoid function in Spearman
    # 2018 ('s') that determines uncertainty in player arrival time
    params['kappa_def'] = kappa_def  # kappa parameter in Spearman 2018 (=1.72 in the paper)
    # that gives the advantage defending players to control ball, I have set to 1 so that home & away players have
    # same ball control probability
    params['lambda_att'] = lambda_att  # ball control parameter for attacking team
    params['lambda_def'] = lambda_att * params['kappa_def']  # ball control parameter for defending team
    params['lambda_gk'] = params['lambda_def'] * kappa_gk  # make goal keepers must quicker to control ball (because
    # they can catch it)
    params['average_ball_speed'] = abs  # average ball travel speed in m/s
    # numerical parameters for model evaluation
    params['int_dt'] = dt  # integration timestep (dt)
    params['max_int_time'] = mit  # upper limit on integral time
    params['model_converge_tol'] = model_converge_tol  # assume convergence when PPCF>0.99 at a given location.
    # The following are 'short-cut' parameters. We do not need to calculated PPCF explicitly when a player has a
    # sufficient head start. A sufficient head start is when a player arrives at the target location at least
    # 'time_to_control' seconds before the next player
    params['time_to_control_att'] = time_to_control_veto * np.log(10) * (
            np.sqrt(3) * params['tti_sigma'] / np.pi + 1 / params['lambda_att'])
    params['time_to_control_def'] = time_to_control_veto * np.log(10) * (
            np.sqrt(3) * params['tti_sigma'] / np.pi + 1 / params['lambda_def'])
    return params


class player:
    def __init__(self, data, pid, GK, team, params=None,frame=None):
        self.id = str(pid)
        self.org_data = data
        self.team = team
        self.GK = GK
        self.frame = frame
        if params is None:
            self.params = default_model_params()
        else:
            self.params = params
        self.player_name = '_'.join([self.team, self.id])
        self.position, self.inframe = self.get_position()
        self.velocity = self.get_velocity()
        self.time_to_intercept = None
        self.PPCF = 0.

    def __str__(self):
        if self.frame is None:
            return f'Player {self.id} playing for the {self.team} team.'
        else:
            return f'Player {self.id} playing for the {self.team} team. Data for frame {self.frame}. '

    def get_position(self):
        position = np.array([self.org_data[self.player_name + '_x'], self.org_data[self.player_name + '_y']])
        inframe = not np.any(np.isnan(position))
        return position, inframe

    def get_velocity(self):
        velocity = np.array([self.org_data[self.player_name + '_vx'], self.org_data[self.player_name + '_vy']])
        if np.any(np.isnan(velocity)):
            velocity = np.array([0., 0.])
        return velocity

    def simple_time_to_intercept(self, r_final):
        self.PPCF = 0.  # initialise this for later; from what I can tell this is not necessary here but let's keep it

        # Time to intercept assumes that the player continues moving at current velocity for 'reaction_time' seconds
        # and then runs at full speed to the target position.
        r_reaction = self.position + self.velocity * self.params['reaction_time']

        # time to intercept is the reaction time and the time needed for the straight line between position after
        # reaction time and interception location
        # np.linalg.norm returns norm matrix of second order / euclidean norm
        # equating to the direct distance / hypotenuse of pythagoras
        self.time_to_intercept = self.params['reaction_time'] + np.linalg.norm(r_final - r_reaction) / self.params['max_player_speed']

        return self.time_to_intercept

    def probability_intercept_ball(self, T, include_time_to_intercept_calc=False, r_final=None):
        if include_time_to_intercept_calc:
            if r_final is not None:
                self.simple_time_to_intercept(r_final)
            raise ValueError('r_final invalid!')

        # probability of a player arriving at target location at time 'T' or earlier given their expected
        # time_to_intercept (time of arrival), as described in Spearman 2017 eq 3
        f = 1 / (1. + np.exp(-np.pi / np.sqrt(3.0) / self.params['tti_sigma'] * (T - self.time_to_intercept)))
        return f


def pitch_control_at_frame(frame, td_object, n_grid_cells_x=50, offside=False, attacking_team='Home', params=None):
    if params is None:
        params = default_model_params()
    data = td_object.data
    x_range = abs(td_object.x_range_pitch[0] - td_object.x_range_pitch[1])
    y_range = abs(td_object.y_range_pitch[0] - td_object.y_range_pitch[1])

    # get current ball position
    ball_start_pos = data.filter(regex=fr'ball_')
    ball_start_pos = ball_start_pos.loc[frame]

    n_grid_cells_y = int(n_grid_cells_x * y_range / x_range)

    # alternatively mesh like this for more flexibility:
    # https://floodlight.readthedocs.io/en/latest/_modules/floodlight/models/space.html#DiscreteVoronoiModel

    # grid units
    dx = x_range / n_grid_cells_x
    dy = y_range / n_grid_cells_y

    xgrid = np.arange(n_grid_cells_x) * dx + dx / 2
    ygrid = np.arange(n_grid_cells_y) * dy + dx / 2

    # initialise pitch control grids for attacking and defending teams
    PPCFa = np.zeros(shape=(len(ygrid), len(xgrid)))
    PPCFd = np.zeros(shape=(len(ygrid), len(xgrid)))

    # get the players for both teams and sort by attacking and defending (formality)
    if attacking_team == 'Home':
        attacking_players = get_all_players(td_object=td_object, frame=frame, teams=['Home'], params=params)
        defending_players = get_all_players(td_object=td_object, frame=frame, teams=['Away'], params=params)
    elif attacking_team == 'Away':
        attacking_players = get_all_players(td_object=td_object, frame=frame, teams=['Away'], params=params)
        defending_players = get_all_players(td_object=td_object, frame=frame, teams=['Home'], params=params)
    else:
        raise ValueError('team must be either "Home" or "Away"!')

    if offside:
        # check offside with function to be included later
        print('Offside check not available yet.')

    # calculate pitch pitch control model at each location on the pitch
    for i in range(len(ygrid)):
        for j in range(len(xgrid)):
            target_position = np.array([xgrid[j], ygrid[i]])
            PPCFa[i, j], PPCFd[i, j] = pitch_control_at_target(target_position, attacking_players, defending_players,
                                                               ball_start_pos, params)
    # check probability sums within convergence
    checksum = np.sum(PPCFa + PPCFd) / float(n_grid_cells_y * n_grid_cells_x)
    assert 1 - checksum < params['model_converge_tol'], "Checksum failed: %1.3f" % (1 - checksum)
    return PPCFa, xgrid, ygrid


def pitch_control_at_target(target_position, attacking_players, defending_players, ball_start_pos, params=None):
    if params is None:
        params = default_model_params()

    # calculate ball travel time from start position to end position.
    if ball_start_pos is None or any(np.isnan(ball_start_pos)):  # assume that ball is already at location
        ball_travel_time = 0.0
    else:
        # ball travel time is distance to target position from current ball position divided assumed average ball speed
        ball_travel_time = np.linalg.norm(target_position - ball_start_pos) / params['average_ball_speed']

    # first get arrival time of 'nearest' attacking player (nearest also dependent on current velocity)
    tau_min_att = np.nanmin([p.simple_time_to_intercept(target_position) for p in attacking_players])
    tau_min_def = np.nanmin([p.simple_time_to_intercept(target_position) for p in defending_players])

    # check whether we actually need to solve equation 3
    if tau_min_att - max(ball_travel_time, tau_min_def) >= params['time_to_control_def']:
        # if defending team can arrive significantly before attacking team, no need to solve pitch control model
        return 0., 1.
    elif tau_min_def - max(ball_travel_time, tau_min_att) >= params['time_to_control_att']:
        # if attacking team can arrive significantly before defending team, no need to solve pitch control model
        return 1., 0.
    else:
        # solve pitch control model by integrating equation 3 in Spearman et al.
        # first remove any player that is far (in time) from the target location
        attacking_players = [p for p in attacking_players if
                             p.time_to_intercept - tau_min_att < params['time_to_control_att']]
        defending_players = [p for p in defending_players if
                             p.time_to_intercept - tau_min_def < params['time_to_control_def']]
        # set up integration arrays
        dT_array = np.arange(ball_travel_time - params['int_dt'], ball_travel_time + params['max_int_time'],
                             params['int_dt'])
        PPCFatt = np.zeros_like(dT_array)
        PPCFdef = np.zeros_like(dT_array)

        # integration equation 3 of Spearman 2018 until convergence or tolerance limit hit (see 'params')
        ptot = 0.0
        i = 1
        while 1 - ptot > params['model_converge_tol'] and i < dT_array.size:
            T = dT_array[i]
            for player in attacking_players:
                # calculate ball control probablity for 'player' in time interval T+dt
                dPPCFdT = (1 - PPCFatt[i - 1] - PPCFdef[i - 1]) * player.probability_intercept_ball(
                    T) * player.params['lambda_att']
                # make sure it's greater than zero
                assert dPPCFdT >= 0, 'Invalid attacking player probability (calculate_pitch_control_at_target)'
                player.PPCF += dPPCFdT * params['int_dt']  # total contribution from individual player
                PPCFatt[
                    i] += player.PPCF  # add to sum over players in the attacking team (remembering array element is zero at the start of each integration iteration)
            for player in defending_players:
                # calculate ball control probablity for 'player' in time interval T+dt
                dPPCFdT = (1 - PPCFatt[i - 1] - PPCFdef[i - 1]) * player.probability_intercept_ball(
                    T) * player.params['lambda_def']
                # make sure it's greater than zero
                assert dPPCFdT >= 0, 'Invalid defending player probability (calculate_pitch_control_at_target)'
                player.PPCF += dPPCFdT * params['int_dt']  # total contribution from individual player
                PPCFdef[i] += player.PPCF  # add to sum over players in the defending team
            ptot = PPCFdef[i] + PPCFatt[i]  # total pitch control probability
            i += 1
        if i >= dT_array.size:
            print("Integration failed to converge: %1.3f" % (ptot))
        return PPCFatt[i - 1], PPCFdef[i - 1]


def plot_pitch_control(td_object, frame, attacking_team='Home', PPCF=None, velocities=False, params=None, n_grid_cells_x=50):

    if PPCF is None:
        PPCF, xgrid, ygrid = pitch_control_at_frame(frame, td_object, params=params, n_grid_cells_x=n_grid_cells_x)

    fig, ax = td_object.plot_players(frame=frame, velocities=velocities)

    if attacking_team == 'Home':
        cmap = 'bwr'
    else:
        cmap = 'brw_r'
    ax.imshow(np.flipud(PPCF), extent=(min(td_object.x_range_pitch), max(td_object.x_range_pitch), min(td_object.y_range_pitch), max(td_object.y_range_pitch)), cmap=cmap, alpha=0.5, vmin=0.0, vmax=1.0)
    return fig, ax
