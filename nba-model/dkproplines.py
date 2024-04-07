#Imports
import re
import urllib.request
from bs4 import BeautifulSoup
import json
from itertools import permutations
import numpy as np

#Read Data in From DraftKings
url = "https://www.scoresandodds.com/nba/props"
request = urllib.request.Request(url)
html = urllib.request.urlopen(request).read()

#Create JSON 
for line in data_lines:
    json_str = line.strip()
    json_dict = json.loads(json_str)

#Data About Points, Rebounds & Assists Subcategory of Player Props Stored Here
category = json_dict['eventGroup']['offerCategories'][1]['offerSubcategoryDescriptors'][3]['offerSubcategory']['offers']
#Number of Games for the Day That You Can Bet on. Make into a List to Loop Through
games = np.arange(0, len(category)).tolist()

#Create List for Number of Players
num_players = []

#Get Number of Players in Each Game and Append Player Counts to num_players list
for players in games:
    num_players.append(len(category[players]))

#Create a Range for the Number of Player Per Game
#For example - 7 Players in a Game would be [0,1,2,3,4,5,6]
num_players_range = []

#Create Range of Players for Each Game
for x in num_players:
    num_players_range.append(np.arange(0, x).tolist())

#Reset games and num_players_range to x and y     
x = games
y = num_players_range

#Create a list that holds the combination of games and players to loop through
#This will be used to get the player information by game from category
game_combo = [(i,j) for i in x for j in y[i]]
#List that will hold the player information dictionary 
player_info = []

#For each game and player combintation in game_combo append the player data to player_info
for x,y in game_combo:  
     player_info.append(category[x][y]['outcomes'])

#Create a list with the total number of players across both games
total_num_players = np.arange(0, len(player_info)).tolist()

#Create lists that will store data for each players lines and odds
participant = []
line = []
odds = []

#For each player in total_num_players, get the over and under (0,1) with the player's name, label, and oddsAmerican
for x in total_num_players:
    for y in (0,1):
        participant.append(player_info[x][y]['participant'])
        line.append(player_info[x][y]['label'])
        odds.append(player_info[x][y]['oddsAmerican'])

#Imports
import pandas as pd

#PRA = Points, Rebounds, Assists Table
PRA = pd.DataFrame({'PLAYER': participant,
                             'LINE': line,
                             'ODDS': odds})

#Create DataFrame
#Group by Player - Aggregate Since There are Two Rows For Each Player - One for Over and One for Under
PRA_Grouped = PRA.groupby('PLAYER').agg(lambda x: x.tolist())

print(PRA_Grouped)