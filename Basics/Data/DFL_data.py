# Data handling for dfl data
# Data not publicly available!

import sys

sys.path.append('C:\\Users\\DavidB\\PycharmProjects\\My_Football_Analytics')
import pandas as pd
import numpy as np
from floodlight.io import dfl
from lxml import etree
import flatdict
import datetime


# -------------------------------------------------------------------------------------------
# Function to parse dfl position data xml to data frame (compatible with tracking data class)
# -------------------------------------------------------------------------------------------

# requires position data and match information file relies on some functions from the floodlight.io module (dfl) -
# https://floodlight.readthedocs.io/en/latest/modules/io/dfl.html
def parse_dfl_pos_data(pos_filepath, mi_filepath, fps=25, print_info=False):
    # get both teams' teamsheets
    teamsheets = dfl.read_teamsheets_from_mat_info_xml(mi_filepath)

    # get link between player id, kit number and team
    links_jID_to_xID = {
        "Home": teamsheets['Home'].get_links("jID", "tID"),
        "Away": teamsheets['Away'].get_links("jID", "tID"),
    }
    links_pID_to_jID = {
        "Home": teamsheets['Home'].get_links("pID", "jID"),
        "Away": teamsheets['Away'].get_links("pID", "jID"),
    }

    # initialize data as dictionary with relevant lists = columns
    data = dict()
    data['ball'] = dict()
    data['ball']['x'] = []
    data['ball']['y'] = []
    data['ballstatus'] = []
    data['possession'] = []
    data['GameSection'] = []
    data['Time'] = []
    data['Time [s]'] = []

    # dummy dates with time to get time variable
    firsthalf_time = datetime.datetime(100, 1, 1, 0, 0, 0, 0)
    secondhalf_time = datetime.datetime(100, 1, 1, 0, 45, 0, 0)
    time_s = 0
    # using floodlight function to get periods and the associated "N"s i.e frame names (not starting at 0!) from
    # position data
    period_frames, est_framerate = dfl._create_periods_from_dat(pos_filepath)

    # every FrameSet (one per player and half!)
    for e, (_, frame_set) in enumerate(etree.iterparse(pos_filepath, tag="FrameSet")):

        # get ball position
        segment = frame_set.get("GameSection")  # firstHalf or secondHalf | constant in FrameSet!
        if print_info:
            print(segment)
        # get ball position
        if frame_set.get("TeamId").lower() == "ball":  # the ball has its own TeamId ("ball")
            for frame in frame_set:  # every frame = actual frame with coordinates, ball status and possession
                data['ball']['x'].append(float(frame.get('X')))
                data['ball']['y'].append(float(frame.get('Y')))
                data['ballstatus'].append(float(frame.get("BallStatus")))
                data['possession'].append(float(frame.get("BallPossession")))
                data['GameSection'].append(frame_set.get("GameSection"))
                # add time column always starting at 0 and adding 0.04 seconds per frame (25fps)
                # starting at 45 minutes if segement is secondHalf
                ms_per_frame = 1000 / fps
                data['Time [s]'].append(time_s)
                time_s = time_s + ms_per_frame / 1000
                if segment == 'firstHalf':
                    data['Time'].append(firsthalf_time.time().strftime('%H:%M:%S.%f'))
                    firsthalf_time += datetime.timedelta(milliseconds=ms_per_frame)
                elif segment == 'secondHalf':
                    data['Time'].append(secondhalf_time.time().strftime('%H:%M:%S.%f'))
                    secondhalf_time += datetime.timedelta(milliseconds=ms_per_frame)
        # get player positions
        else:
            # all frames in frame set
            frames = [frame for frame in frame_set.iterfind("Frame")]

            # frame number where player's data starts and ends
            sf_player = int(frames[0].get("N"))
            ef_player = int(frames[-1].get("N"))

            # the correct index for data in general depends on the half. The start index is either 0 or the next
            # index after the last index of the first half the correct end index for the first half is the end
            # frame of the player - the first half overall start frame for the end index we subtract the half's
            # start frame from the end frame and in case of the second half add the frames / indices from first
            # half
            start_index = 0 if segment == 'firstHalf' else period_frames['firstHalf'][1] - \
                                                           period_frames['firstHalf'][0] + 1
            end_index = period_frames['firstHalf'][1] - period_frames['firstHalf'][0] if segment == 'firstHalf' else \
                period_frames['secondHalf'][1] - period_frames['secondHalf'][0] + start_index

            # players available position data has to start at the following index: if half == 1 it's just the
            # player's start frame minus the overall start frame (+ start index (=0)) if half == 2 player's start
            # frame - half's start frame + start_index (= indices of first half) for the second half we subtract
            # the second halfs start frame from the player's end frame (index number played in second half) and
            # add the start_index as the number of indices / frames contained in first half
            start_index_player = sf_player - period_frames['firstHalf'][
                0] if segment == 'firstHalf' else sf_player - period_frames['secondHalf'][0] + start_index
            end_index_player = ef_player - period_frames['firstHalf'][0] if segment == 'firstHalf' else ef_player - \
                                                                                                        period_frames[
                                                                                                            'secondHalf'][
                                                                                                            0] + start_index
            if print_info:
                print(frame_set.get('PersonId'))
                print(f'Players first and last frame: {sf_player, ef_player}')  # his first frame
                print(f'As indices this equates to: {start_index_player, end_index_player}')
                print(
                    f'Overall the data for this half starts end ends with the following indices: {start_index, end_index}')

            # home team player?
            if frame_set.get("PersonId") in links_pID_to_jID["Home"]:
                jrsy = links_pID_to_jID['Home'][frame_set.get("PersonId")]  # get kit number from player ID

                # only for the first time we need to initialize, then it already exists (for second half)
                if f'Home_{jrsy}' not in data.keys():
                    data[f'Home_{jrsy}'] = dict()
                    data[f'Home_{jrsy}']['x'] = []
                    data[f'Home_{jrsy}']['y'] = []
                    # if player starts in second half we need to add nans for first half as well
                    if segment == 'secondHalf':
                        data[f'Home_{jrsy}']['x'] += list(np.repeat(np.nan, start_index))
                        data[f'Home_{jrsy}']['y'] += list(np.repeat(np.nan, start_index))

                # add as many nan at the start as required by difference in start indices
                data[f'Home_{jrsy}']['x'] += list(np.repeat(np.nan, start_index_player - start_index))
                data[f'Home_{jrsy}']['y'] += list(np.repeat(np.nan, start_index_player - start_index))

                # add the given data
                data[f'Home_{jrsy}']['x'] += [float(frame.get("X")) for frame in frames]
                data[f'Home_{jrsy}']['y'] += [float(frame.get("Y")) for frame in frames]

                # add as many nan at the end as required by difference in start indices
                data[f'Home_{jrsy}']['x'] += list(np.repeat(np.nan, end_index - end_index_player))
                data[f'Home_{jrsy}']['y'] += list(np.repeat(np.nan, end_index - end_index_player))

            # away team player?
            elif frame_set.get("PersonId") in links_pID_to_jID["Away"]:
                jrsy = links_pID_to_jID['Away'][frame_set.get("PersonId")]  # get kit number from player ID

                # only for the first time we need to intitalize, then it already exists (for second half)
                if f'Away_{jrsy}' not in data.keys():
                    data[f'Away_{jrsy}'] = dict()
                    data[f'Away_{jrsy}']['x'] = []
                    data[f'Away_{jrsy}']['y'] = []
                    # if player starts in second half we need to add nans for first half as well
                    if segment == 'secondHalf':
                        data[f'Away_{jrsy}']['x'] += list(np.repeat(np.nan, start_index))
                        data[f'Away_{jrsy}']['y'] += list(np.repeat(np.nan, start_index))
                # add as many nan at the start as required by difference in start indeces
                data[f'Away_{jrsy}']['x'] += list(np.repeat(np.nan, start_index_player - start_index))
                data[f'Away_{jrsy}']['y'] += list(np.repeat(np.nan, start_index_player - start_index))
                # add the given data
                data[f'Away_{jrsy}']['x'] += [float(frame.get("X")) for frame in frames]
                data[f'Away_{jrsy}']['y'] += [float(frame.get("Y")) for frame in frames]

                # add as many nan at the end as required by difference in start indeces
                data[f'Away_{jrsy}']['x'] += list(np.repeat(np.nan, end_index - end_index_player))
                data[f'Away_{jrsy}']['y'] += list(np.repeat(np.nan, end_index - end_index_player))

    flat_data = flatdict.FlatDict(data, delimiter='_')
    flat_data.keys()
    df = pd.DataFrame.from_dict(flat_data, orient='index').transpose()
    period_dict = {'firstHalf': 1,
                   'secondHalf': 2}
    df['Period'] = df['GameSection'].map(period_dict)

    pitch = dfl.read_pitch_from_mat_info_xml(mi_filepath)
    x_range_data = pitch.xlim
    y_range_data = pitch.ylim
    return df, x_range_data, y_range_data
