import pandas as pd
import numpy as np
from Tracking_Data import tracking_data  # wrong here but necessary for use in notebooks


# class player to bundle all steps for each player

def get_player_from_data(td_object, pid, data=None, frame=None):
    if data is None:
        data = td_object.data
    GK = str(pid) in [td_object.Home_GK, td_object.Away_GK]
    if frame is None:
        player_data = data.filter(regex=fr'_{pid}_|Period|Time')
        team = player_data.columns[2].split('_')[0]
        p_object = player(data=player_data, pid=pid, GK=GK, team=team, frame=frame)
    else:
        player_data = data.filter(regex=fr'_{pid}_|Period|Time')
        player_data = player_data.loc[frame]
        team = player_data.keys()[2].split('_')[0]
        p_object = player(data=player_data, pid=pid, GK=GK, team=team, frame=frame)
    return p_object


def get_all_players(td_object, frame=None, teams=['Home', 'Away']):
    data = td_object.data
    player_ids = np.unique([c.split('_')[1] for c in data.keys() if c[:4] in teams])
    players = []
    for p in player_ids:
        c_player = get_player_from_data(td_object=td_object, pid=p, frame=frame)
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
    # sufficient head start. A sufficient head start is when the a player arrives at the target location at least
    # 'time_to_control' seconds before the next player
    params['time_to_control_att'] = time_to_control_veto * np.log(10) * (
                np.sqrt(3) * params['tti_sigma'] / np.pi + 1 / params['lambda_att'])
    params['time_to_control_def'] = time_to_control_veto * np.log(10) * (
                np.sqrt(3) * params['tti_sigma'] / np.pi + 1 / params['lambda_def'])
    return params


class player:
    def __init__(self, data, pid, GK, team, frame=None, vmax=5, reaction_time=0.7,
                 tti_sigma=0.45, kappa_def=1, lambda_att=4.3, kappa_gk=0.7):
        self.id = str(pid)
        self.org_data = data
        self.team = team
        self.GK = GK
        self.frame = frame
        self.vmax = vmax  # player max speed in m/s. Could be individualised
        self.reaction_time = reaction_time  # player reaction time in 's'. Could be individualised
        self.tti_sigma = tti_sigma  # standard deviation of sigmoid function (see Eq 3 in Spearman, 2017)
        self.lambda_att = lambda_att  # control rate parameter ~ time it takes the player to control the
        # ball (see Eq 4 in Spearman, 2017)
        self.lambda_def = lambda_att * kappa_gk if self.GK else lambda_att * kappa_def
        #  ensures that anything near the GK is likely to be claimed by the GK
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
        # Time to intercept assumes that the player continues moving at current velocity for 'reaction_time' seconds
        # and then runs at full speed to the target position.
        r_reaction = self.position + self.velocity * self.reaction_time

        # time to intercept is the reaction time and the time needed for the straight line between position after
        # reaction time and interception location
        # np.linalg.norm returns norm matrix of second order / euclidean norm
        # equating to the direct distance / hypotenuse of pythagoras
        self.time_to_intercept = self.reaction_time + np.linalg.norm(r_final - r_reaction) / self.vmax

        return self.time_to_intercept

    def probability_intercept_ball(self, T, r_final):
        self.simple_time_to_intercept(r_final)
        # probability of a player arriving at target location at time 'T' or earlier given their expected
        # time_to_intercept (time of arrival), as described in Spearman 2017 eq 3
        f = 1 / (1. + np.exp(-np.pi / np.sqrt(3.0) / self.tti_sigma * (T - self.time_to_intercept)))
        return f
