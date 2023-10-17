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


def get_all_players(td_object, frame=None, teams = ['Home', 'Away']):

    data = td_object.data
    player_ids = np.unique([c.split('_')[1] for c in data.keys() if c[:4] in teams])
    players = []
    for p in player_ids:
        c_player = get_player_from_data(td_object=td_object, pid=p, frame=frame)
        if c_player.inframe: # to check data is not nan i.e. on the pitch
            players.append(c_player)
    return players



class player:
    def __init__(self, data, pid, GK, team, frame=None, vmax=5, reaction_time=0.7,
                 tti_sigma=0.45, kappa_def=1, lambda_att=4.3, kappa_gk=0.7):
        self.id = str(pid)
        self.org_data = data
        self.team = team
        self.GK = GK
        self.frame = frame
        self.vmax = vmax                        # player max speed in m/s. Could be individualised
        self.reaction_time = reaction_time      # player reaction time in 's'. Could be individualised
        self.tti_sigma = tti_sigma              # standard deviation of sigmoid function (see Eq 3 in Spearman, 2017)
        self.lambda_att = lambda_att            # control rate parameter ~ time it takes the player to control the
                                                # ball (see Eq 4 in Spearman, 2017)
        self.lambda_def = lambda_att*kappa_gk if self.GK else lambda_att*kappa_def
                                                #  ensures that anything near the GK is likely to be claimed by the GK
        self.player_name = '_'.join([self.team, self.id])
        self.position, self.inframe = self.get_position()

    def __str__(self):
        if self.frame is None:
            return f'Player {self.id} playing for the {self.team} team.'
        else:
            return f'Player {self.id} playing for the {self.team} team. Data for frame {self.frame}. '

    def get_position(self):
        position = np.array([self.org_data[self.player_name+'_x'], self.org_data[self.player_name+'_y']])
        inframe = not np.any(np.isnan(position))
        return position, inframe

"""
# time to intercept

position = np.array([10, 10])
velocity = np.array([5, 5])
reaction_time = 0.7
r_final = np.array([10, 15])
vmax = 10

# position after running in the same direction for reaction time =  position + velocity multiplied by reaction time
r_reaction = position + velocity * reaction_time
# time to intercept is the reaction time and the time needed for the straight line between position after reaction time and interception location
# np.linalg.norm returns norm matrix of second order / euclidean norm equating to the direct distance / hypotenuse of pythagoras
time_to_intercept = reaction_time + np.linalg.norm(r_final - r_reaction) / vmax
time_to_intercept
"""
