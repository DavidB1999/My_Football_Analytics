import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import time


# -----------------------------------------------------
# all shots and related statistics for a specific match
# # -----------------------------------------------------

def scrape_match_shots(match_id='23092'):
    base_url = 'https://understat.com/match/'
    match = str(match_id)
    url = base_url + match

    res = requests.get(url)

    soup = BeautifulSoup(res.content, 'lxml')

    scripts = soup.find_all('script')

    strings = scripts[1].string

    ind_start = strings.index("('") + 2
    ind_end = strings.index("'),")
    json_data = strings[ind_start:ind_end]
    json_data = json_data.encode('utf8').decode('unicode_escape')

    data = json.loads(json_data)

    x = []
    y = []
    minute = []
    xG = []
    player = []
    result = []
    team = []

    data_away = data['a']
    data_home = data['h']

    for index in range(len(data_home)):  # indexes are h and a
        x += [data_home[index][key] for key in data_home[index] if key == 'X']
        y += [data_home[index][key] for key in data_home[index] if key == 'Y']
        xG += [data_home[index][key] for key in data_home[index] if key == 'xG']
        minute += [data_home[index][key] for key in data_home[index] if key == 'minute']
        team += [data_home[index][key] for key in data_home[index] if key == 'h_team']
        player += [data_home[index][key] for key in data_home[index] if key == 'player']
        result += [data_home[index][key] for key in data_home[index] if key == 'result']

    for index in range(len(data_away)):  # indexes are h and a
        x += [data_away[index][key] for key in data_away[index] if key == 'X']
        y += [data_away[index][key] for key in data_away[index] if key == 'Y']
        xG += [data_away[index][key] for key in data_away[index] if key == 'xG']
        minute += [data_away[index][key] for key in data_away[index] if key == 'minute']
        team += [data_away[index][key] for key in data_away[index] if key == 'a_team']
        player += [data_away[index][key] for key in data_away[index] if key == 'player']
        result += [data_away[index][key] for key in data_away[index] if key == 'result']

    # create the actual data frame
    col_names = ['x', 'y', 'xG', 'minute', 'team', 'player', 'result']
    df = pd.DataFrame([x, y, xG, minute, team, player, result], index=col_names)
    df = df.T
    return df


# ---------------------------
# team season shot statistics
# ---------------------------

def scrape_team_Data(team='Union_Berlin', season='2023', by='situation', out_as='df'):
    base_url = 'https://understat.com/team/'
    url = base_url + team + '/' + season

    res = requests.get(url)

    soup = BeautifulSoup(res.content, 'lxml')

    scripts = soup.find_all('script')

    stData = scripts[2].string  # var statisticsData

    ind_start = stData.index("('") + 2
    ind_end = stData.index("');")
    json_data = stData[ind_start:ind_end]
    json_data = json_data.encode('utf8').decode('unicode_escape')
    data = json.loads(json_data)

    # unpack the dictionary for data "against"
    for key in data[by]:
        for key2 in data[by][key]['against']:
            data[by][key]['_'.join([key2, 'against'])] = data[by][key]['against'][key2]
        del data[by][key]['against']

    if out_as == 'df':
        out_data = pd.DataFrame.from_dict(data[by], orient='index')
    elif out_as == 'dict':
        out_data = data[by]
    else:
        raise ValueError(
            f'out_as should be either "df" for a pandas dataframe of "dict" for a dictionary. {out_as} is not an option!')

    return out_data


# --------------------------------------
# shot data by player for an entire team
# --------------------------------------

def scrape_player_Data(team='Union_Berlin', season='2023', out_as='df'):
    base_url = 'https://understat.com/team/'
    url = base_url + team + '/' + season

    res = requests.get(url)

    soup = BeautifulSoup(res.content, 'lxml')

    scripts = soup.find_all('script')

    stData = scripts[3].string  # var statisticsData

    ind_start = stData.index("('") + 2
    ind_end = stData.index("');")
    json_data = stData[ind_start:ind_end]
    json_data = json_data.encode('utf8').decode('unicode_escape')
    data = json.loads(json_data)

    if out_as == 'df':
        out_data = pd.DataFrame.from_dict(data)

    elif out_as == 'dict':
        out_data = data
    else:
        raise ValueError(
            f'out_as should be either "df" for a pandas dataframe of "dict" for a dictionary. {out_as} is not an option!')

    return out_data
