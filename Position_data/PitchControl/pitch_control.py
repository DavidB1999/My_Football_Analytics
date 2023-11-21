import pandas as pd
import numpy as np
from Tracking_Data import tracking_data  # wrong here but necessary for use in notebooks
from Pitch.My_Pitch import \
    myPitch  # might need adaptation of path depending on whether it is used in pycharm or jupyter notebook
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from torch.nn.functional import softplus


# class player to bundle all steps for each player
def get_player_from_data(td_object, pid, team, data=None, frame=None, params=None):
    if params is None:
        params = default_model_params()
    if data is None:
        data = td_object.data
    GK = str(pid) == td_object.Home_GK if team == 'Home' else str(pid) == td_object.Away_GK
    if frame is None:
        player_data = data.filter(regex=fr'{team}_{pid}_|Period|Time')
        team = player_data.columns[2].split('_')[0]
        p_object = player(data=player_data, pid=pid, GK=GK, team=team, params=params, frame=frame)
    else:
        player_data = data.filter(regex=fr'{team}_{pid}_|Period|Time')
        player_data = player_data.loc[frame]
        team = player_data.keys()[2].split('_')[0]
        p_object = player(data=player_data, pid=pid, GK=GK, team=team, params=params, frame=frame)
    return p_object


def get_all_players(td_object, frame=None, teams=['Home', 'Away'], params=None):
    if params is None:
        params = default_model_params()
    data = td_object.data
    players = []
    for team in teams:
        player_ids = np.unique([c.split('_')[1] for c in data.keys() if c[:4] == team])
        for p in player_ids:
            c_player = get_player_from_data(td_object=td_object, pid=p, frame=frame, params=params, team=team)
            if c_player.inframe:  # to check data is not nan i.e. on the pitch
                players.append(c_player)
    return players


def check_offside(td_object, frame, attacking_team, verbose=False, tol=0.2):
    # get players
    attacking_players = get_all_players(td_object=td_object, frame=frame, teams=[attacking_team])
    defending_team = ['Home', 'Away']
    defending_team.remove(attacking_team)
    defending_players = get_all_players(td_object=td_object, frame=frame, teams=defending_team)

    # get ball position
    ball_x = td_object.data['ball_x'][frame]

    # find jersey number of defending goalkeeper (just to establish attack direction)
    defending_GK_id = td_object.Home_GK if attacking_team == 'Away' else td_object.Away_GK

    # make sure defending goalkeeper is actually on the field!
    assert defending_GK_id in [p.id for p in
                               defending_players], "Defending goalkeeper jersey number not found in defending players"

    # get goalkeeper player object
    defending_GK = [p for p in defending_players if p.id == defending_GK_id][0]

    # use defending goalkeeper x position to figure out which half he is defending
    # distance to both goal lines!
    defending_half = td_object.x_range_pitch[0] if abs(defending_GK.position[0] - td_object.x_range_pitch[0]) < abs(
        defending_GK.position[0] - td_object.x_range_pitch[1]) else td_object.x_range_pitch[1]

    # find the x-position of the second-deepest defeending player (including GK)
    # reverse depending on attacking direction
    r = True if defending_half == max(td_object.x_range_pitch) else False
    second_deepest_defender_x = sorted([p.position[0] for p in defending_players], reverse=r)[1]

    # define offside line as being the maximum/minumum of second_deepest_defender_x, ball position and half-way line
    # max vs min depends on direction of play so we just use r again!
    if r:
        offside_line = max(second_deepest_defender_x, ball_x,
                           0.5 * max(td_object.x_range_pitch) - min(td_object.x_range_pitch)) + tol
    else:
        offside_line = min(second_deepest_defender_x, ball_x,
                           0.5 * max(td_object.x_range_pitch) - min(td_object.x_range_pitch)) - tol

    # any attacking players with x-position greater/smaller than the offside line are offside
    if verbose:
        for p in attacking_players:
            if r:
                if p.position[0] > offside_line:
                    print("player %s in %s team is offside" % (p.id, p.player_name))
            else:
                if p.position[0] < offside_line:
                    print("player %s in %s team is offside" % (p.id, p.player_name))
    if r:
        attacking_players = [p for p in attacking_players if p.position[0] <= offside_line]
    else:
        attacking_players = [p for p in attacking_players if p.position[0] >= offside_line]

    return attacking_players


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
    def __init__(self, data, pid, GK, team, params=None, frame=None):
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
        if self.frame is None:
            position = np.array([self.org_data[self.player_name + '_x'].astype('float'),
                                 self.org_data[self.player_name + '_y'].astype('float')])
        else:
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
        self.time_to_intercept = self.params['reaction_time'] + np.linalg.norm(r_final - r_reaction) / self.params[
            'max_player_speed']

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
    ball_start_pos = ball_start_pos.loc[frame].astype('float')

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
        if offside:
            attacking_players = check_offside(td_object=td_object, frame=frame, attacking_team='Home')
        else:
            attacking_players = get_all_players(td_object=td_object, frame=frame, teams=['Home'], params=params)
        defending_players = get_all_players(td_object=td_object, frame=frame, teams=['Away'], params=params)
    elif attacking_team == 'Away':
        attacking_players = get_all_players(td_object=td_object, frame=frame, teams=['Away'], params=params)
        if offside:
            defending_players = check_offside(td_object=td_object, frame=frame, attacking_team='Away')
        else:
            defending_players = get_all_players(td_object=td_object, frame=frame, teams=['Home'], params=params)
    else:
        raise ValueError('team must be either "Home" or "Away"!')

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
        # include only those players that arrive (time to intercept) earlier than the fastest teammate + the time it would take him to control
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


def plot_pitch_control(td_object, frame, attacking_team='Home', PPCF=None, velocities=False, params=None,
                       n_grid_cells_x=50, offside=False):
    if PPCF is None:
        PPCF, xgrid, ygrid = pitch_control_at_frame(frame, td_object, params=params, n_grid_cells_x=n_grid_cells_x,
                                                    offside=offside, attacking_team=attacking_team)

    fig, ax = td_object.plot_players(frame=frame, velocities=velocities)

    if attacking_team == 'Home':
        cmap = 'bwr'
    else:
        cmap = 'bwr_r'
    ax.imshow(np.flipud(PPCF), extent=(
        min(td_object.x_range_pitch), max(td_object.x_range_pitch), min(td_object.y_range_pitch),
        max(td_object.y_range_pitch)), cmap=cmap, alpha=0.5, vmin=0.0, vmax=1.0)
    return fig, ax


def animate_pitch_control(td_object, start_frame, end_frame, attacking_team='Home', velocities=False, params=None,
                          n_grid_cells_x=50, frames_per_second=None, fname='Animated_Clip', pitch_col='#1c380e',
                          line_col='white', colors=['red', 'blue', 'black'], PlayerAlpha=0.7, fpath=None,
                          progress_steps=[0.25, 0.5, 0.75], offside=False):
    if frames_per_second is None:
        frames_per_second = td_object.fps
    data = td_object.data
    if start_frame == 0:
        data = data.iloc[start_frame: end_frame]
    else:
        data = data.iloc[start_frame - 1: end_frame]
    index = data.index
    index_range = end_frame - start_frame

    if progress_steps is not None:
        progress_dict = dict()
        for p in progress_steps:
            progress_dict[p] = False

    field_dimen = (max(td_object.dimensions['x']['pitch']), max(td_object.dimensions['y']['pitch']))

    if fpath is not None:
        fname = fpath + '/' + fname + '.mp4'  # path and filename
    else:
        fname = fname + '.mp4'

    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Pitch Control animation', artist='Matplotlib',
                    comment=f'{td_object.data_source} based pitch control clip')
    writer = FFMpegWriter(fps=frames_per_second, metadata=metadata)

    # create pitch
    if td_object.scale_to_pitch == 'mplsoccer':
        pitch = Pitch(pitch_color=pitch_col, line_color=line_col)
        fig, ax = plt.subplots()
        fig.set_facecolor(pitch_col)
        pitch.draw(ax=ax)
    elif td_object.scale_to_pitch == 'myPitch':
        pitch = myPitch(grasscol=pitch_col)
        fig, ax = plt.subplots()  # figsize=(13.5, 8)
        fig.set_facecolor(pitch_col)
        pitch.plot_pitch(ax=ax)
    else:
        raise ValueError(f'Unfortunately the pitch {td_object.scale_to_pitch} is not yet supported by this function!')

    fig.set_tight_layout(True)
    print("Generating your clip...")  # , end=''

    # determine colormap
    if attacking_team == 'Home':
        cmap = 'bwr'
    else:
        cmap = 'bwr_r'

    with writer.saving(fig, fname, 100):
        for i in index:
            figobjs = []  # this is used to collect up all the axis objects so that they can be deleted after each iteration
            # for both teams
            PPCF, xgrid, ygrid = pitch_control_at_frame(i, td_object, params=params, n_grid_cells_x=n_grid_cells_x,
                                                        offside=offside, attacking_team=attacking_team)
            pc = ax.imshow(np.flipud(PPCF), extent=(
                min(td_object.x_range_pitch), max(td_object.x_range_pitch), min(td_object.y_range_pitch),
                max(td_object.y_range_pitch)), cmap=cmap, alpha=0.5, vmin=0.0, vmax=1.0)
            figobjs.append(pc)
            for team, color in zip(['home', 'away', 'ball'], colors):
                # get x and y values
                x_values = data[td_object.dimensions[td_object.x_cols_pattern][''.join([team, '_columns'])]].loc[
                    i].astype('float')
                y_values = data[td_object.dimensions[td_object.y_cols_pattern][''.join([team, '_columns'])]].loc[
                    i].astype('float')
                objs = ax.scatter(x=x_values, y=y_values, s=20, c=color)
                figobjs.append(objs)

                if velocities and team != 'ball':
                    vx_columns = ['{}_vx'.format(c[:-2]) for c in
                                  list(td_object.dimensions[td_object.x_cols_pattern][''.join(
                                      [team, '_columns'])])]  # column header for player x positions
                    vy_columns = ['{}_vy'.format(c[:-2]) for c in
                                  list(td_object.dimensions[td_object.y_cols_pattern][''.join(
                                      [team, '_columns'])])]  # column header for player y positions
                    objs = ax.quiver(x_values, y_values, data[vx_columns].loc[i], data[vy_columns].loc[i],
                                     color=color, angles='xy', scale_units='xy', scale=1, width=0.0015,
                                     headlength=5, headwidth=3, alpha=PlayerAlpha)
                    figobjs.append(objs)
            frame_minute = int(data[td_object.time_col][i] / 60.)
            frame_second = (data[td_object.time_col][i] / 60. - frame_minute) * 60.
            timestring = "%d:%1.2f" % (frame_minute, frame_second)
            objs = ax.text(field_dimen[0] / 2 - 0.05 * field_dimen[0],
                           td_object.y_range_pitch[1] + 0.05 * td_object.y_range_pitch[1],
                           timestring, fontsize=14)
            figobjs.append(objs)
            writer.grab_frame()
            # Delete all axis objects (other than pitch lines) in preparation for next frame
            for figobj in figobjs:
                figobj.remove()

            if progress_steps is not None:
                for key in progress_dict.keys():
                    if (i - start_frame) / index_range >= key and not progress_dict[key]:
                        print(f'{key * 100}% done!')
                        progress_dict[key] = True
                        break

    print("All done!")
    plt.clf()
    plt.close(fig)


# function to create pitch control model via tensors as done by anenglishgoat but using my tracking data class
def tensor_pitch_control(td_object, jitter=1e-12, pos_nan_to=-1000, vel_nan_to=0, remove_first_frames=0,
                         reaction_time=0.7, max_player_speed=5, average_ball_speed=15, sigma=0.45, lamb=4.3,
                         n_grid_points_x=50, n_grid_points_y=30, device='cpu', dtype=torch.float32,
                         first_frame=0, last_frame=500, batch_size=250, deg=50, version='GL', max_int=500,
                         team='Home'):
    # create the entire exponent of the intercept probability function based on sigma
    exp = np.pi / np.sqrt(3.) / sigma

    # make sure velocities are computed
    td_object.get_velocities()

    # access position data
    Home = td_object.get_team('Home', selection='position', T_P=False)
    Away = td_object.get_team('Away', selection='position', T_P=False)
    Ball = td_object.get_ball(pos_only=True)

    # access velocities
    vel_Home = td_object.get_team('Home', selection='velocity', T_P=False)
    vel_Away = td_object.get_team('Away', selection='velocity', T_P=False)

    # convert to arrays
    Ball_array = pos_to_array(Ball, nan_to=np.nan, ball=True)
    Home_array = pos_to_array(Home, nan_to=pos_nan_to)
    Away_array = pos_to_array(Away, nan_to=pos_nan_to)
    Home_vel_array = pos_to_array(vel_Home, nan_to=vel_nan_to) + jitter
    Away_vel_array = pos_to_array(vel_Away, nan_to=vel_nan_to) + jitter

    # reshaping for tensor later
    Home_array = Home_array[:, remove_first_frames:, None, None, :]
    Away_array = Away_array[:, remove_first_frames:, None, None, :]
    Home_vel_array = Home_vel_array[:, remove_first_frames:, None, None, :]
    Away_vel_array = Away_vel_array[:, remove_first_frames:, None, None, :]
    Ball_array = Ball_array[None, remove_first_frames:]

    # create grid based on tensors
    XX, YY = torch.meshgrid(torch.linspace(td_object.x_range_pitch[0], td_object.x_range_pitch[1], n_grid_points_x,
                                           device=device, dtype=dtype),
                            torch.linspace(td_object.y_range_pitch[0], td_object.y_range_pitch[1], n_grid_points_y,
                                           device=device, dtype=dtype))
    target_position = torch.stack([XX, YY], 2)[None, None, :, :, :]  # all possible positions

    # number of frames over which we model pitch control
    n_frames = last_frame - first_frame

    # get number of players
    n_players = Home_array.shape[0] + Away_array.shape[0]
    h_players = Home_array.shape[0]
    # time to intercept empty torch
    tti = torch.empty([n_players, batch_size, n_grid_points_x, n_grid_points_y], device=device, dtype=dtype)
    # n_players*batch_size*grid
    tmp2 = torch.empty([n_players, batch_size, n_grid_points_x, n_grid_points_y, 1], device=device, dtype=dtype)
    # n_players*batch_size*grid
    pc = torch.empty([n_frames, n_grid_points_x, n_grid_points_y], device=device, dtype=dtype)
    # frames * grid

    if version == 'GL':
        print("Running Spearman's pitch control computation based on Gauss legendre quadration")

        ti, wi = np.polynomial.legendre.leggauss(deg=deg)  ## used for numerical integration later on
        ti = torch.tensor(ti, device=device, dtype=dtype)
        wi = torch.tensor(wi, device=device, dtype=dtype)

        # loop over batches needed to cover all frames
        for b in range(int(n_frames / batch_size)):
            print(f'Current batch: {b + 1}/{int(n_frames / batch_size)}')
            # convert all arrays to tensors!
            bp = torch.tensor(
                Ball_array[:, (first_frame + b * batch_size):(np.minimum(first_frame + (b + 1) * batch_size,
                                                                         int(first_frame + n_frames)))],
                device=device, dtype=dtype)
            hp = torch.tensor(
                Home_array[:, (first_frame + b * batch_size):(np.minimum(first_frame + (b + 1) * batch_size,
                                                                         int(first_frame + n_frames)))],
                device=device, dtype=dtype)
            ap = torch.tensor(
                Away_array[:, (first_frame + b * batch_size):(np.minimum(first_frame + (b + 1) * batch_size,
                                                                         int(first_frame + n_frames)))],
                device=device, dtype=dtype)
            hv = torch.tensor(
                Home_vel_array[:, (first_frame + b * batch_size):(np.minimum(first_frame + (b + 1) * batch_size,
                                                                             int(first_frame + n_frames)))],
                device=device, dtype=dtype)
            av = torch.tensor(
                Away_vel_array[:, (first_frame + b * batch_size):(np.minimum(first_frame + (b + 1) * batch_size,
                                                                             int(first_frame + n_frames)))],
                device=device, dtype=dtype)

            ball_travel_time = torch.norm(target_position - bp, dim=4).div_(average_ball_speed)

            r_reaction_home = hp + hv.mul_(reaction_time)  # position after reaction time (vector)
            r_reaction_away = ap + av.mul_(reaction_time)  # = position + velocity multiplied by reaction time
            r_reaction_home = r_reaction_home - target_position  # distance to target position (vector)
            r_reaction_away = r_reaction_away - target_position  # after reaction time

            # time to intercept for home and away filled

            tti[:h_players, :ball_travel_time.shape[1]] = torch.norm(r_reaction_home, dim=4).add_(reaction_time).div_(
                max_player_speed)
            tti[h_players:, :ball_travel_time.shape[1]] = torch.norm(r_reaction_away, dim=4).add_(reaction_time).div_(
                max_player_speed)

            tmp2[..., 0] = exp * (ball_travel_time - tti)
            tmp1 = exp * 0.5 * (ti + 1) * 10 + tmp2
            if team == 'Home':
                hh = torch.sigmoid(tmp1[:h_players]).mul_(lamb)
            elif team == 'Away':
                hh = torch.sigmoid(tmp1[h_players:]).mul_(lamb)
            else:
                raise ValueError(f'team needs to be either "Home" or "Away". {team} is not valid')

            h = hh.sum(0)

            S = torch.exp(-lamb * torch.sum(softplus(tmp1) - softplus(tmp2), dim=0).div_(exp))
            # fill pitch control tensor
            pc[(0 + b * batch_size):(
                np.minimum(0 + (b + 1) * batch_size, int(0 + n_frames)))] = torch.matmul(S * h, wi).mul_(5.)

    elif version == 'int':
        print("Running pitch control computation based on Spearman's integration method")
        relu = torch.nn.ReLU()
        dt = 1 / td_object.fps
        # loop over batches needed to cover all frames
        for b in range(int(n_frames / batch_size)):
            print(f'Current batch: {b + 1}/{int(n_frames / batch_size)}')
            # convert all arrays to tensors!
            bp = torch.tensor(
                Ball_array[:, (first_frame + b * batch_size):(np.minimum(first_frame + (b + 1) * batch_size,
                                                                         int(first_frame + n_frames)))],
                device=device, dtype=dtype)
            hp = torch.tensor(
                Home_array[:, (first_frame + b * batch_size):(np.minimum(first_frame + (b + 1) * batch_size,
                                                                         int(first_frame + n_frames)))],
                device=device, dtype=dtype)
            ap = torch.tensor(
                Away_array[:, (first_frame + b * batch_size):(np.minimum(first_frame + (b + 1) * batch_size,
                                                                         int(first_frame + n_frames)))],
                device=device, dtype=dtype)
            hv = torch.tensor(
                Home_vel_array[:, (first_frame + b * batch_size):(np.minimum(first_frame + (b + 1) * batch_size,
                                                                             int(first_frame + n_frames)))],
                device=device, dtype=dtype)
            av = torch.tensor(
                Away_vel_array[:, (first_frame + b * batch_size):(np.minimum(first_frame + (b + 1) * batch_size,
                                                                             int(first_frame + n_frames)))],
                device=device, dtype=dtype)

            ball_travel_time = torch.norm(target_position - bp, dim=4).div_(average_ball_speed)

            r_reaction_home = hp + hv.mul_(reaction_time)  # position after reaction time (vector)
            r_reaction_away = ap + av.mul_(reaction_time)  # = position + velocity multiplied by reaction time
            r_reaction_home = r_reaction_home - target_position  # distance to target position (vector)
            r_reaction_away = r_reaction_away - target_position  # after reaction time

            # time to intercept for home and away filled

            tti[:h_players, :ball_travel_time.shape[1]] = torch.norm(r_reaction_home, dim=4).add_(reaction_time).div_(
                max_player_speed)
            tti[h_players:, :ball_travel_time.shape[1]] = torch.norm(r_reaction_away, dim=4).add_(reaction_time).div_(
                max_player_speed)

            y = torch.zeros([n_players, bp.shape[1], n_grid_points_x, n_grid_points_y], device=device, dtype=dtype)
            for tt in range(max_int):
                sumy = torch.sum(y, dim=0)  # control over all players
                if torch.min(sumy) > 0.99:  # convergence
                    break
                # added relu to tackle infinite negative due to infinite large sumy!
                y += dt * lamb * relu(1. - sumy) * 1. / (1. + torch.exp(-exp * (dt * tt + ball_travel_time - tti)))

            if team == 'Home':
                pc[(first_frame + b * batch_size):
                   (np.minimum(first_frame + (b + 1) * batch_size, int(first_frame + n_frames)))] = y[:h_players].sum(0)
            elif team == 'Away':
                pc[(first_frame + b * batch_size):
                   (np.minimum(first_frame + (b + 1) * batch_size, int(first_frame + n_frames)))] = y[h_players:].sum(0)
            else:
                raise ValueError(f'team needs to be either "Home" or "Away". {team} is not valid')

    return pc


def plot_tensor_pitch_control(td_object, frame, pitch_control=None, jitter=1e-12, pos_nan_to=-1000, vel_nan_to=0,
                              remove_first_frames=0, reaction_time=0.7, max_player_speed=5, average_ball_speed=15,
                              sigma=0.45, lamb=4.3, n_grid_points_x=50, n_grid_points_y=30, device='cpu',
                              dtype=torch.float32, first_frame=0, last_frame=500, batch_size=250, deg=50, version='GL',
                              max_int=500, cmap=None, velocities=True, flip_y=True, team='Home'):
    if pitch_control is None:
        pitch_control = tensor_pitch_control(td_object=td_object, jitter=jitter, pos_nan_to=pos_nan_to,
                                             vel_nan_to=vel_nan_to, remove_first_frames=remove_first_frames,
                                             reaction_time=reaction_time, max_player_speed=max_player_speed,
                                             average_ball_speed=average_ball_speed, sigma=sigma, lamb=lamb,
                                             n_grid_points_x=n_grid_points_x, n_grid_points_y=n_grid_points_y,
                                             device=device,
                                             dtype=dtype, first_frame=first_frame, last_frame=last_frame,
                                             batch_size=batch_size, deg=deg, version=version, max_int=max_int)

    # determine colormap
    if cmap is None:
        if team == 'Home':
            cmap = 'bwr'
        elif team == 'Away':
            cmap = 'bwr_r'
        else:
            raise ValueError(f'team needs to be either "Home" or "Away". {team} is not valid')

    frame_number = frame - first_frame
    # plot players
    fig, ax = td_object.plot_players(frame=frame_number, velocities=velocities)

    # ensure correct orientation
    mx_pitch = max(td_object.y_range_pitch)
    mx_data = max(td_object.y_range_data)
    if flip_y:
        ax.imshow(np.flipud(pitch_control[frame_number].rot90()), extent=(
            td_object.x_range_pitch[0], td_object.x_range_pitch[1], td_object.y_range_pitch[1],
            td_object.y_range_pitch[0]), cmap=cmap, alpha=0.5, vmin=0.0, vmax=1.0)
    else:
        ax.imshow(np.flipud(pitch_control[frame_number].rot90()), extent=(
            td_object.x_range_pitch[0], td_object.x_range_pitch[1], td_object.y_range_pitch[0],
            td_object.y_range_pitch[1]), cmap=cmap, alpha=0.5, vmin=0.0, vmax=1.0)

    return fig


# convert data frame to array (usually for position data in pitch control model
def pos_to_array(pos_data, nan_to, ball=False):
    if 'Period' in pos_data.columns or 'Time [s]' in pos_data.columns:
        raise ValueError('Data should include positon data only. Not any other columns!')
    n_players = int(len(pos_data.columns) / 2)
    if ball:
        array = np.asarray(pos_data.iloc[:, range(0, 2)])[:, None, None, :]
    else:
        array = np.array([np.asarray(pos_data.iloc[:, range(j * 2, j * 2 + 2)]) for j in range(n_players)])

    np.nan_to_num(array, copy=False, nan=nan_to)
    return array


def animate_tensor_pitch_control(td_object, pitch_control=None, jitter=1e-12, pos_nan_to=-1000, vel_nan_to=0,
                                 remove_first_frames=0, reaction_time=0.7, max_player_speed=5, average_ball_speed=15,
                                 sigma=0.45, lamb=4.3, n_grid_points_x=50, n_grid_points_y=30, device='cpu',
                                 dtype=torch.float32, first_frame_calc=0, last_frame_calc=500, batch_size=250, deg=50,
                                 version='GL', max_int=500, cmap=None, velocities=True, flip_y=True, team='Home',
                                 progress_steps=[0.25, 0.5, 0.75], frames_per_second=None, fpath=None,
                                 fname='Animation', pitch_col='#1c380e', line_col='white',
                                 colors=['red', 'blue', 'black'], PlayerAlpha=0.7, first_frame_ani=0,
                                 last_frame_ani=100):
    if frames_per_second is None:
        frames_per_second = td_object.fps

    # get position data
    field_dimen = (max(td_object.dimensions['x']['pitch']), max(td_object.dimensions['y']['pitch']))
    data = td_object.data
    if first_frame_ani == 0:
        data = data.iloc[first_frame_ani: last_frame_ani]
    else:
        data = data.iloc[first_frame_ani - 1: last_frame_ani]
    index = data.index
    index_range = last_frame_ani - first_frame_ani

    # determine colormap
    if cmap is None:
        if team == 'Home':
            cmap = 'bwr'
        elif team == 'Away':
            cmap = 'bwr_r'
        else:
            raise ValueError(f'team needs to be either "Home" or "Away". {team} is not valid')

    # get pitch control for entire frame!?
    if pitch_control is None:
        pitch_control = tensor_pitch_control(td_object=td_object, jitter=jitter, pos_nan_to=pos_nan_to,
                                             vel_nan_to=vel_nan_to, remove_first_frames=remove_first_frames,
                                             reaction_time=reaction_time, max_player_speed=max_player_speed,
                                             average_ball_speed=average_ball_speed, sigma=sigma, lamb=lamb,
                                             n_grid_points_x=n_grid_points_x, n_grid_points_y=n_grid_points_y,
                                             device=device,
                                             dtype=dtype, first_frame=first_frame_calc, last_frame=last_frame_calc,
                                             batch_size=batch_size, deg=deg, version=version, max_int=max_int)

    if progress_steps is not None:
        progress_dict = dict()
        for p in progress_steps:
            progress_dict[p] = False

    # define path
    if fpath is not None:
        fname = fpath + '/' + fname + '.mp4'  # path and filename
    else:
        fname = fname + '.mp4'

    # set up writer
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Pitch Control animation', artist='Matplotlib',
                    comment=f'{td_object.data_source} based pitch control clip')
    writer = FFMpegWriter(fps=frames_per_second, metadata=metadata)

    # create pitch
    if td_object.scale_to_pitch == 'mplsoccer':
        pitch = Pitch(pitch_color=pitch_col, line_color=line_col)
        fig, ax = plt.subplots()
        fig.set_facecolor(pitch_col)
        pitch.draw(ax=ax)
    elif td_object.scale_to_pitch == 'myPitch':
        pitch = myPitch(grasscol=pitch_col)
        fig, ax = plt.subplots()  # figsize=(13.5, 8)
        fig.set_facecolor(pitch_col)
        pitch.plot_pitch(ax=ax)
    else:
        raise ValueError(f'Unfortunately the pitch {td_object.scale_to_pitch} is not yet supported by this function!')

    fig.set_tight_layout(True)
    print("Generating your clip...")  # , end=''

    with writer.saving(fig, fname, 100):
        for i in index:
            figobjs = []  # this is used to collect up all the axis objects so that they can be deleted after each iteration
            if flip_y:
                PC = ax.imshow(np.flipud(pitch_control[i - 1 - first_frame_ani].rot90()), extent=(
                    td_object.x_range_pitch[0], td_object.x_range_pitch[1], td_object.y_range_pitch[1],
                    td_object.y_range_pitch[0]), cmap=cmap, alpha=0.5, vmin=0.0, vmax=1.0)
            else:
                PC = ax.imshow(np.flipud(pitch_control[i - 1 - first_frame_ani].rot90()), extent=(
                    td_object.x_range_pitch[0], td_object.x_range_pitch[1], td_object.y_range_pitch[0],
                    td_object.y_range_pitch[1]), cmap=cmap, alpha=0.5, vmin=0.0, vmax=1.0)
            figobjs.append(PC)
            for team, color in zip(['home', 'away', 'ball'], colors):
                # get x and y values
                x_values = data[td_object.dimensions[td_object.x_cols_pattern][''.join([team, '_columns'])]].loc[
                    i].astype('float')
                y_values = data[td_object.dimensions[td_object.y_cols_pattern][''.join([team, '_columns'])]].loc[
                    i].astype('float')
                objs = ax.scatter(x=x_values, y=y_values, s=20, c=color)
                figobjs.append(objs)
                if velocities and team != 'ball':
                    vx_columns = ['{}_vx'.format(c[:-2]) for c in
                                  list(td_object.dimensions[td_object.x_cols_pattern][''.join(
                                      [team, '_columns'])])]  # column header for player x positions
                    vy_columns = ['{}_vy'.format(c[:-2]) for c in
                                  list(td_object.dimensions[td_object.y_cols_pattern][''.join(
                                      [team, '_columns'])])]  # column header for player y positions
                    objs = ax.quiver(x_values, y_values, data[vx_columns].loc[i], data[vy_columns].loc[i],
                                     color=color, angles='xy', scale_units='xy', scale=1, width=0.0015,
                                     headlength=5, headwidth=3, alpha=PlayerAlpha)
                    figobjs.append(objs)

            frame_minute = int(data[td_object.time_col][i] / 60.)
            frame_second = (data[td_object.time_col][i] / 60. - frame_minute) * 60.
            timestring = "%d:%1.2f" % (frame_minute, frame_second)
            objs = ax.text(field_dimen[0] / 2 - 0.05 * field_dimen[0],
                           td_object.y_range_pitch[1] + 0.05 * td_object.y_range_pitch[1],
                           timestring, fontsize=14)
            figobjs.append(objs)
            writer.grab_frame()
            # Delete all axis objects (other than pitch lines) in preparation for next frame
            for figobj in figobjs:
                figobj.remove()

            if progress_steps is not None:
                for key in progress_dict.keys():
                    if (i - first_frame_ani) / index_range >= key and not progress_dict[key]:
                        print(f'{key * 100}% done!')
                        progress_dict[key] = True
                        break
    print("All done!")
    plt.clf()
    plt.close(fig)
