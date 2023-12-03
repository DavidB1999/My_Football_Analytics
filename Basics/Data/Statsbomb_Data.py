import sys

sys.path.append('C:\\Users\\DavidB\\PycharmProjects\\My_Football_Analytics')
import pandas as pd
import numpy as np
from statsbombpy import sb

# -------------------------------------------------------------
# checkout https://github.com/statsbomb/statsbombpy/tree/master
# -------------------------------------------------------------


# valid event types
valid_events = ['starting_xis', 'half_starts', 'camera ons', 'camera offs', 'passes', 'ball_receipts',
                'carrys', 'pressures', 'foul_committeds', 'foul_wons', 'duels', 'interceptions', 'blocks',
                'referee_ball_drops', 'ball_recoverys', 'dispossesseds', 'clearances', 'dribbles',
                'miscontrols', 'shots', 'goal_keepers', 'dribbled_pasts', 'injury_stoppages', 'half_ends',
                'substitutions', 'shields', 'tactical_shifts', 'own_goal_againsts', 'own_goal_fors',
                'bad_behaviours', 'player_offs', 'player_ons', '50/50s', 'errors', 'offsides']


def free_data():
    comps = sb.competitions()
    comps = comps[~comps['match_available_360'].isnull()]
    return comps


# ------------------
# events for a match
# ------------------
def match_events(match_id, event_filter=None, team_column='team', return_teams=True):
    # all events for the match
    if event_filter is None:
        events = sb.events(match_id=match_id)
    elif event_filter not in valid_events:
        raise ValueError(f'event filter needs to be one of {valid_events}. You supplied {event_filter}!')
    else:
        events = sb.events(match_id=match_id, split=True, flatten_attrs=False)[event_filter]

    if return_teams:
        # get home and away team from lineup order!
        xis = sb.events(match_id=match_id, split=True, flatten_attrs=False)['starting_xis']
        teams = list(xis[team_column])
        return events, teams
    else:
        return events


# --------------------------------
# events for an entire competition
# --------------------------------

# problems with runtime error and multiple processing (not in notebooks but in pycharm)
def competition_events(country, division, season, gender, event_filter=None):
    # all events for the match
    events = None
    events_dict = None
    if event_filter is None:
        if __name__ == '__main__':
            events = sb.competition_events(country=country, division=division, season=season, gender=gender)
        return events
    elif event_filter not in valid_events:
        raise ValueError(f'event filter needs to be one of {valid_events}. You supplied {event_filter}!')
    else:
        if __name__ == '__main__':
            events_dict = sb.competition_events(country=country, division=division, season=season, split=True)
            events = events_dict[event_filter]
        return events, events_dict


events, event_dict = competition_events(country="International",
                                        division="FIFA World Cup",
                                        season="2022",
                                        gender="male", event_filter='shots')

print(events)
