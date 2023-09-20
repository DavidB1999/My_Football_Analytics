from Create_Radar import create_radar_plotly, param_select
import plotly
from Basics.Data.FBref_PlayerData import Scrape_Player_via_Link

df1, player1, pos1 = Scrape_Player_via_Link('https://fbref.com/en/players/2c0558b8/Jamal-Musiala')
df2, player2, pos2 = Scrape_Player_via_Link('https://fbref.com/en/players/74618572/Kaoru-Mitoma')

params = ['xAG', 'Shot-Creating Actions', 'Pass Completion %', 'Non-Penalty xG', 'Progressive Passes',
          'Progressive Carries', 'Successful Take-Ons', 'Successful Take-On %']
df1 = param_select(df1, params, 'Variables')
df2 = param_select(df2, params, 'Variables')

radar_chart = create_radar_plotly(df1=df1, player1=player1, df2=df2, player2=player2, legend=True)

plotly.offline.plot(radar_chart)