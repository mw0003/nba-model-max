# NBA PLAYER PROJECTION MODEL

# API https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/static/players.md

#career stats
from calendar import c
from queue import Empty
from nba_api.stats.endpoints import playercareerstats, playergamelog, teamestimatedmetrics,playerdashptshotdefend, commonplayerinfo, teamplayeronoffdetails, leaguelineupviz
from nba_api.stats.static import players
from nba_api.stats.static import teams
from nba_api.stats.library.parameters import RunType

import requests, pandas
from bs4 import BeautifulSoup

# Get all of Todays Games
from datetime import datetime, timezone
from dateutil import parser
from nba_api.live.nba.endpoints import scoreboard
from nba_api.live.nba.endpoints import boxscore


from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import cumestatsteamgames, cumestatsteam, gamerotation, matchupsrollup,winprobabilitypbp,boxscoreadvancedv3, playervsplayer, boxscoreplayertrackv3, teamandplayersvsplayers
import pandas as pd
import numpy as np
import json
import difflib
import time
import requests

from nn import *

from runtype import Dispatch
dp = Dispatch()

#TODO : PLayers on/off details, Rest days details

#import sklearn
from scipy import stats
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn import preprocessing, svm
import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import math
import scipy
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing 
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import sklearn
import warnings
 
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

import json

#Function to get injured players and starting lineups from scraping
def getLineups():
    #Get Starting lineups 
    url = "https://www.rotowire.com/basketball/nba-lineups.php"
    soup = BeautifulSoup(requests.get(url).text, "html.parser")

    #starters class
    lineups = soup.find_all(class_='is-pct-play-100')
    positions = [x.find('div').text for x in lineups]
    names = [x.find('a')['title'] for x in lineups]
    teams = sum([[x.text] * 5 for x in soup.find_all(class_='lineup__abbr')], [])

    print(names)
    for name in names:
        
        p = players.find_players_by_full_name(name)
        print(p)

    df = pandas.DataFrame(zip(names, teams, positions))
    jsonLineups = df.to_json(orient='split')


    #Get injured Players 
    playersOut = soup.find_all(class_='is-pct-play-0')
    names2 = [x.find('a')['title'] for x in playersOut]

    df = pandas.DataFrame(zip(names2))

    json_object = df.to_json(orient='split')
    print(json_object)


    #Write out/in to file

    with open('lineups.json', 'w', encoding='utf-8') as f:
        json.dump(jsonLineups, f, ensure_ascii=False, indent=4)

    with open('playersOut.json', 'w', encoding='utf-8') as f:
        json.dump(json_object, f, ensure_ascii=False, indent=4)

#gets todays game dicts
def getTodaysGames():
    f = "{gameId}: {awayTeam} vs. {homeTeam} @ {gameTimeLTZ}" 
    board = scoreboard.ScoreBoard()
    print("ScoreBoardDate: " + board.score_board_date)
    games = board.games.get_dict()
    for game in games:
        gameTimeLTZ = parser.parse(game["gameTimeUTC"]).replace(tzinfo=timezone.utc).astimezone(tz=None)
        print(f.format(gameId=game['gameId'], awayTeam=game['awayTeam']['teamName'], homeTeam=game['homeTeam']['teamName'], gameTimeLTZ=gameTimeLTZ))
    return games

#gets lineups from files
def readInLineups():
    with open('lineups.json', 'r') as f:
        todaysLineups = json.load(f)
        todaysLineups = json.loads(todaysLineups)
        #print(todaysLineups)
    df = pandas.DataFrame(todaysLineups['data'], columns=todaysLineups['columns'])
    with open('playersOut.json', 'r') as f:
        injured = json.load(f)
        injured = json.loads(injured)
        #print(injured)
    df2 = pandas.DataFrame(injured['data'], columns=injured['columns'])

    #df2 = pandas.read_json('lineups.json', orient='split')
   # print(df)

    return df, df2

def nonlinearProjection(player_game_log, col):

    df = player_game_log[[
            "SEASON_ID",
            "Player_ID",
            "Game_ID",
            "GAME_DATE",
            "MATCHUP",
            "WL",
            "MIN",
            "FGM",
            "FGA",
            "FG_PCT",
            "FG3M",
            "FG3A",
            "FG3_PCT",
            "FTM",
            "FTA",
            "FT_PCT",
            "OREB",
            "DREB",
            "REB",
            "AST",
            "STL",
            "BLK",
            "TOV",
            "PF",
            "PTS",
            "PLUS_MINUS",
            "VIDEO_AVAILABLE"
        ]]
    forecast_col = col
    forecast_out = int(math.ceil(0.01 * len(df)))
    print("out: ", forecast_out)
    df['label'] = df[forecast_col]

    print(df)
    print("DF LABEL: ", df['label'])
    df2 = df['label']
    X = np.array(range(len(df2))).reshape(-1, 1)

    print(X)
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    #X = X[:-forecast_out]
    print("LATELY", X_lately)

    df.dropna(inplace=True)

    y = np.array(df['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)

    regressor = RandomForestRegressor()
  #  clf = LinearRegression(n_jobs=-1)
   # clf.fit(X_train, y_train)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)
    conf = regressor.score(X_test, y_test)

    print(predictions)
    print(conf)

    #print( X_train, X_test, y_train, y_test)
   # confidence = clf.score(X_test, y_test)

    #print("confidence Score: ", confidence)
   # forecast_set = clf.predict(X_lately)
   # print("projection: ", forecast_set)
   # df['Forecast'] = np.nan

   # last_date = df.iloc[-1].name
  #  print(last_date)
  #  next_date = last_date+1
   # for i in forecast_set:
   #     df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
   #     next_date += 1
   #     print(next_date)
    return predictions


    # Assuming df is your DataFrame
   # X = df.iloc[:,1:2].values  #features
   # y = df.iloc[:,col].values  # Target variable
   # label_encoder = LabelEncoder()
   # x_categorical = df.select_dtypes(include=['object']).apply(label_encoder.fit_transform)
   # x_numerical = df.select_dtypes(exclude=['object']).values
   # x = pd.concat([pd.DataFrame(x_numerical), x_categorical], axis=1).values

    # Fitting Random Forest Regression to the dataset
   # regressor = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)

    # Fit the regressor with x and y data
  #  regressor.fit(x, y)
   # oob_score = regressor.oob_score_
   # print(f'Out-of-Bag Score: {oob_score}')
    
    # Making predictions on the same data or new data
  #  predictions = regressor.predict(x)
    
    # Evaluating the model
  #  mse = mean_squared_error(y, predictions)
  #  print(f'Mean Squared Error: {mse}')
    
   # r2 = r2_score(y, predictions)
  #  print(f'R-squared: {r2}')

  #  print("Predictions: ", predictions)


def lineupViz():
    df = leaguelineupviz.LeagueLineupViz().get_data_frames()
    print(df)

def teamvplayerOnOff():
    df = teamandplayersvsplayers.TeamAndPlayersVsPlayers().get_data_frames()
    print(df)

def defensiveStats(playerId, oppTeam):
    test = playerdashptshotdefend.PlayerDashPtShotDefend(player_id=playerId,
        team_id='0').get_data_frames()

    print(test)
    #sys.sleep()
    return test[0]
#Linear Regression for track boxscores
def linearProjectionTrackBox(df1, colname):

    df = df1[['reboundChancesTotal', 'touches', 'passes','assists','contestedFieldGoalsMade','contestedFieldGoalPercentage','contestedFieldGoalsAttempted', 'uncontestedFieldGoalsMade','uncontestedFieldGoalsPercentage', 'uncontestedFieldGoalsAttempted', 'defendedAtRimFieldGoalsMade','defendedAtRimFieldGoalPercentage', 'defendedAtRimFieldGoalsAttempted']]
    forecast_col = colname
    forecast_out = int(math.ceil(0.01 * len(df)))
    print("out: ", forecast_out)
    df['label'] = df[forecast_col]

    print(df)
    print("DF LABEL: ", df['label'])
    df2 = df['label']
    X = np.array(range(len(df2))).reshape(-1, 1)

    print(X)
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    #X = X[:-forecast_out]
    print("LATELY", X_lately)

    df.dropna(inplace=True)

    y = np.array(df['label'])

    if len(X) <= 0 or len(y) <=0:
        return None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)
    print( X_train, X_test, y_train, y_test)
    confidence = clf.score(X_test, y_test)

    print("confidence Score: ", confidence)
    forecast_set = clf.predict(X_lately)
    print("projection: ", forecast_set)
    df['Forecast'] = np.nan

    last_date = df.iloc[-1].name
    print(last_date)
    next_date = last_date+1
    for i in forecast_set:
        df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
        next_date += 1
        print(next_date)
    return forecast_set


#on/off summary

def onOffDF():
    df = teamplayeronoffdetails.TeamPlayerOnOffDetails().get_data_frames()
    print(df)

#projections for player stats by game logs
def linearProjection(player_game_log, colname, loglast):

    df = player_game_log[[
            "SEASON_ID",
            "Player_ID",
            "Game_ID",
            "GAME_DATE",
            "MATCHUP",
            "WL",
            "MIN",
            "FGM",
            "FGA",
            "FG_PCT",
            "FG3M",
            "FG3A",
            "FG3_PCT",
            "FTM",
            "FTA",
            "FT_PCT",
            "OREB",
            "DREB",
            "REB",
            "AST",
            "STL",
            "BLK",
            "TOV",
            "PF",
            "PTS",
            "PLUS_MINUS",
            "VIDEO_AVAILABLE"
        ]]
    dflast = loglast[[
            "SEASON_ID",
            "Player_ID",
            "Game_ID",
            "GAME_DATE",
            "MATCHUP",
            "WL",
            "MIN",
            "FGM",
            "FGA",
            "FG_PCT",
            "FG3M",
            "FG3A",
            "FG3_PCT",
            "FTM",
            "FTA",
            "FT_PCT",
            "OREB",
            "DREB",
            "REB",
            "AST",
            "STL",
            "BLK",
            "TOV",
            "PF",
            "PTS",
            "PLUS_MINUS",
            "VIDEO_AVAILABLE"
        ]]
    df = pandas.concat([dflast, df], ignore_index=True)
    
    forecast_col = colname
    forecast_out = int(math.ceil(0.01 * len(df)))
    print("out: ", forecast_out)
    df['label'] = df[forecast_col]

    print(df)
    print("DF LABEL: ", df['label'])
    df2 = df['label']
    X = np.array(range(len(df2))).reshape(-1, 1)

    print(X)
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    #X = X[:-forecast_out]
    print("LATELY", X_lately)

    df.dropna(inplace=True)

    y = np.array(df['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)
    print( X_train, X_test, y_train, y_test)
    confidence = clf.score(X_test, y_test)

    print("confidence Score: ", confidence)
    forecast_set = clf.predict(X_lately)
    print("projection: ", forecast_set)
    df['Forecast'] = np.nan

    last_date = df.iloc[-1].name
    print(last_date)
    next_date = last_date+1
    for i in forecast_set:
        df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
        next_date += 1
        print(next_date)
    return forecast_set


def linearOneProjection(player_game_log, colname):

    df = player_game_log[[
            "SEASON_ID",
            "Player_ID",
            "Game_ID",
            "GAME_DATE",
            "MATCHUP",
            "WL",
            "MIN",
            "FGM",
            "FGA",
            "FG_PCT",
            "FG3M",
            "FG3A",
            "FG3_PCT",
            "FTM",
            "FTA",
            "FT_PCT",
            "OREB",
            "DREB",
            "REB",
            "AST",
            "STL",
            "BLK",
            "TOV",
            "PF",
            "PTS",
            "PLUS_MINUS",
            "VIDEO_AVAILABLE"
        ]]
    
    forecast_col = colname
    forecast_out = int(math.ceil(0.01 * len(df)))
    print("out: ", forecast_out)
    df['label'] = df[forecast_col]

    print(df)
    print("DF LABEL: ", df['label'])
    df2 = df['label']
    X = np.array(range(len(df2))).reshape(-1, 1)

    print(X)
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    #X = X[:-forecast_out]
    print("LATELY", X_lately)

    df.dropna(inplace=True)

    y = np.array(df['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)
    print( X_train, X_test, y_train, y_test)
    confidence = clf.score(X_test, y_test)

    print("confidence Score: ", confidence)
    forecast_set = clf.predict(X_lately)
    print("projection: ", forecast_set)
    df['Forecast'] = np.nan

    last_date = df.iloc[-1].name
    print(last_date)
    next_date = last_date+1
    for i in forecast_set:
        df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
        next_date += 1
        print(next_date)
    return forecast_set

#conf interval function
def mean_confidence_interval(data, confidence=0.85):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return [m, m-h, m+h]
    
#adv Usg box score
def getUsageBox(Game_id, playerID):
   dfs = boxscoreplayertrackv3.BoxScorePlayerTrackV3(game_id = Game_id)
   df = dfs.get_data_frames()
   playerDFInstance = df[0].loc[df[0]["personId"] == playerID]
  # print(playerDFInstance)
   newDF = playerDFInstance[['position','reboundChancesTotal', 'touches', 'passes','assists','contestedFieldGoalsMade','contestedFieldGoalPercentage','contestedFieldGoalsAttempted', 'uncontestedFieldGoalsMade','uncontestedFieldGoalsPercentage', 'uncontestedFieldGoalsAttempted', 'defendedAtRimFieldGoalsMade','defendedAtRimFieldGoalPercentage', 'defendedAtRimFieldGoalsAttempted']]
   #print(newDF)
   return newDF

#get adv box score by player
def getAdvBoxScore(gameid, playerid):
    advBoxScore = boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=gameid).player_stats
    advBoxScoreDf = advBoxScore.get_data_frame()
    playerDFInstance = advBoxScoreDf.loc[advBoxScoreDf["personId"] == playerid]
   # print(playerDFInstance)
    newDF = playerDFInstance[['minutes', 'assistPercentage', 'assistRatio','reboundPercentage','effectiveFieldGoalPercentage','trueShootingPercentage','usagePercentage','pace','possessions']].copy()
   # print(newDF)
    return newDF


def onOff(playerId, player2Id):
    dic = playervsplayer.PlayerVsPlayer(vs_player_id=player2Id, player_id=playerId).get_data_frames()

    if dic[0].empty or dic[1].empty:
        return None

    overall = dic[0]

    #get individualized rows
    player1row = overall.loc[overall["PLAYER_ID"] == player2Id]
    player2row = overall.loc[overall["PLAYER_ID"] == playerId]

    pminpg = player2row['MIN']/player2row['GP']

    print(pminpg[0])

    onoff = dic[1]
    onRow = onoff.loc[onoff["COURT_STATUS"] == "On"]
    offRow = onoff.loc[onoff["COURT_STATUS"] == "Off"]

    print(onRow)
    print(offRow)
    if offRow.empty or onRow.empty:
        return None

    ppgON = onRow["PTS"].values[0]
    ppgOFF = offRow["PTS"].values[0]

    fgaOn = onRow["FGA"].values[0]
    fgsOff = offRow["FGA"].values[0]

    fg3On = onRow["FG3M"].values[0]
    fg3Off = offRow["FG3M"].values[0]

    gmOff= offRow["MIN"].values[0]
    gmOn = onRow["MIN"].values[0]

    astOn = onRow["AST"].values[0]
    astOff = offRow["AST"].values[0]

    rebOn = onRow["REB"].values[0]
    rebOff = offRow["REB"].values[0]

    stlOn = onRow["STL"].values[0]
    stlOff = offRow["STL"].values[0]

    blkOn = onRow["BLK"].values[0]
    blkOff = offRow["BLK"].values[0]

    gmOn = gmOn/pminpg[0]
    gmOff = gmOff/pminpg[0]

    print(gmOff, gmOn)

    

    ppgON= ppgON/gmOn
    ppgOFF = ppgOFF/gmOff

    fgapgON = fgaOn / gmOn
    fgapgOFF = fgsOff / gmOff

    fg3pgON = fg3On / gmOn
    fg3pgOFF = fg3Off / gmOff

    astOn = astOn/ gmOn
    astOff = astOff / gmOff

    stlOn = stlOn/ gmOn
    stlOff = stlOff / gmOff

    blkOn = blkOn/ gmOn
    blkOff = blkOff / gmOff

    rebOn = rebOn/ gmOn
    rebOff = rebOff / gmOff

    print("ON: ",  fg3pgON,"OFF: ", fg3pgOFF)
    dictionary = {
        "PTSOFF" : ppgOFF,
        "ASTOFF": astOff,
        "FGAOFF": fgapgOFF,
        "3sOFF": fg3pgOFF,
        "REBOFF": rebOff,
        "STLOFF": stlOff,
        "BLKOFF": blkOff
    }
    return dictionary

# MAIN FUNC FOR PLAYER PROJECTIONS
def getPlayerProjection(playerName, team, opposingTeam, opposingTeamLineup, playersOut):
    
    player = players.find_players_by_full_name(playerName)
    print(player[0])

    gameLog2023 = playergamelog.PlayerGameLog(player[0]['id'],2023,'Regular Season')
    gameLog2022 = playergamelog.PlayerGameLog(player[0]['id'],2022,'Regular Season')
    
    #convert playergamelog objects to DFs
    gamesdf2023 = gameLog2023.get_data_frames()[0]
    gamesdf2022 = gameLog2022.get_data_frames()[0]

    if gamesdf2023.empty:
        return "N/A"

    confIntOverall = mean_confidence_interval(gamesdf2023['PTS'])
    confIntOverallAst = mean_confidence_interval(gamesdf2023['AST'])
    confIntOverallReb = mean_confidence_interval(gamesdf2023['REB'])
   
    #Get last 10 games
    lastTen = gamesdf2023.head(15)
    last5  = gamesdf2023.head(5)

    last5pts = last5['PTS'].to_numpy()
    last5ast = last5['AST'].to_numpy()
    last5reb = last5['REB'].to_numpy()
    last5min = last5['MIN'].to_numpy()

    last15pts = lastTen['PTS'].to_numpy()
    last15ast = lastTen['AST'].to_numpy()
    last15reb = lastTen['REB'].to_numpy()
    last15min = lastTen['MIN'].to_numpy()

    #PLAYER HOT/COLD STREAK FINDER
    minGameLog = gamesdf2023['MIN'].to_numpy()
    ptsGameLog = gamesdf2023['PTS'].to_numpy()
    astGameLog = gamesdf2023['AST'].to_numpy()
    rebGameLog = gamesdf2023['REB'].to_numpy()
    

    if len(minGameLog) <1:
        return "N/A"

    minPerGameAvg = sum(minGameLog)/len(minGameLog)
    ptsPerGameAvg = sum(ptsGameLog)/len(ptsGameLog)
    astPerGameAvg = sum(astGameLog)/len(astGameLog)
    rebPerGameAvg = sum(rebGameLog)/len(rebGameLog)

    confIntLast5 = mean_confidence_interval(last5pts)
    confIntLast15 = mean_confidence_interval(last15pts)

    confIntLast5Ast =  mean_confidence_interval(last5ast)
    confIntLast15Ast =  mean_confidence_interval(last15ast)


    confIntLast5Reb =  mean_confidence_interval(last5reb)
    confIntLast15Reb =  mean_confidence_interval(last15reb)


    print(confIntLast5)
    print(confIntLast15)
    print(confIntOverall)

    calcConf = []
    calcConfAst = []
    calcConfReb = []

    #pts 

    floor = (confIntLast15[1] + confIntOverall[1] + confIntLast5[1])/3
    mid = (confIntLast5[0] + confIntLast15[0]+ confIntOverall[0])/3
    top = (confIntLast5[2] + confIntLast15[2]+ confIntOverall[2])/3
    calcConf.append(floor)
    calcConf.append(mid)
    calcConf.append(top)

    #ast

    floor = (confIntLast15Ast[1] + confIntOverallAst[1] + confIntLast5Ast[1])/3
    mid = (confIntLast5Ast[0] + confIntLast15Ast[0]+ confIntOverallAst[0])/3
    top = (confIntLast5Ast[2] + confIntLast15Ast[2]+ confIntOverallAst[2])/3
    calcConfAst.append(floor)
    calcConfAst.append(mid)
    calcConfAst.append(top)

    #reb

    floor = (confIntLast15Reb[1] + confIntOverallReb[1] + confIntLast5Reb[1])/3
    mid = (confIntLast5Reb[0] + confIntLast15Reb[0]+ confIntOverallReb[0])/3
    top = (confIntLast5Reb[2] + confIntLast15Reb[2]+ confIntOverallReb[2])/3
    calcConfReb.append(floor)
    calcConfReb.append(mid)
    calcConfReb.append(top)



    print(calcConf)
    last5minPerGameAvg = sum(last5min)/len(last5min)
    last5ptsPerGameAvg = sum(last5pts)/len(last5pts)
    last5astPerGameAvg = sum(last5ast)/len(last5ast)
    last5rebPerGameAvg = sum(last5reb)/len(last5reb)

    print(last5minPerGameAvg, last5ptsPerGameAvg, last5astPerGameAvg, last5rebPerGameAvg)

    diffmin = last5minPerGameAvg - minPerGameAvg
    last5ptspermin = last5ptsPerGameAvg/last5minPerGameAvg
    ptspermin = ptsPerGameAvg/minPerGameAvg
    last5astpermin = last5astPerGameAvg/last5minPerGameAvg
    astpermin = astPerGameAvg/minPerGameAvg
    last5rebpermin = last5rebPerGameAvg/last5minPerGameAvg
    rebpermin = rebPerGameAvg/minPerGameAvg

    print(minPerGameAvg, last5minPerGameAvg)

    print(ptspermin, last5ptspermin)

    ptsperminDiff = last5ptspermin - ptspermin
    astperminDiff = last5astpermin - astpermin
    rebperminDiff = last5rebpermin - rebpermin

    minHeatIndex = 0
    ptsHeatIndex = 0
    astHeatIndex = 0
    rebHeatIndex = 0

    if ptsperminDiff > 0:
        ptsHeatIndex = ptspermin + ptsperminDiff/2
    elif ptsperminDiff < 0:
        ptsHeatIndex = ptspermin + ptsperminDiff/2
    
    astHeatIndex = astpermin + astperminDiff/2
    rebHeatIndex = rebpermin + rebperminDiff/2
    
    minHeatIndex = minPerGameAvg + diffmin*0.75

    #projections on current heat Indexes
    ptsperminProj = ptsHeatIndex * minHeatIndex
    astperminProj = astHeatIndex * minHeatIndex
    rebperminProj = rebHeatIndex * minHeatIndex
    print(ptsperminProj, astperminProj, rebperminProj)

    if len(last5pts) > 0:

        last5ptsavg = sum(last5pts)/len(last5pts)
        print("LAST 5" , last5ptsavg, type(last5ptsavg))
        last5astavg = sum(last5ast)/len(last5ast)
        last5rebavg = sum(last5reb)/len(last5reb)
    else: 
        last5ptsavg = None
        last5astavg = None
        last5rebavg = None
    

    #gameIds = gamesdf2023["Game_ID"].to_list()
    #print(lastTen)

    if lastTen.empty:
        return
    gameIds = lastTen["Game_ID"].to_numpy()
    playerID = lastTen.iloc[0]["Player_ID"]
    lastTenAdvBoxScores = pd.DataFrame()
    trackBox = pd.DataFrame()

    for id in gameIds:
        time.sleep(0.6)
        df = getAdvBoxScore(id, playerID)
        if df.empty is False:
            lastTenAdvBoxScores = pd.concat([lastTenAdvBoxScores, df])
        df2 = getUsageBox(id, playerID)
        if df2.empty is False:
            trackBox = pd.concat([trackBox, df2])

    print(lastTenAdvBoxScores)
    print(trackBox)

    if len(trackBox.index) < 2:
        return "N/A"

    colsOfinterest = ['reboundChancesTotal', 'touches', 'passes','assists','contestedFieldGoalsMade','contestedFieldGoalsAttempted', 'uncontestedFieldGoalsMade', 'uncontestedFieldGoalsAttempted', 'defendedAtRimFieldGoalsMade', 'defendedAtRimFieldGoalsAttempted']

    #Record projections for player track stats
    trackdict = {}
    for col in colsOfinterest:
        val = linearProjectionTrackBox(trackBox, col)
        if val != None:
            trackdict[col] = val[0]
        else: 
            return "N/A"

   # print(trackdict)
    playerPosition = trackBox['position'].to_numpy()[0]

    if len(playerPosition) < 1:
        playerPosition = "G"
    print("POS: " ,playerPosition[0])
   # print(opposingTeamLineup)

    defenders = []
    oppTeamID = -1
    for p in opposingTeamLineup:
        time.sleep(.600)
        pl = commonplayerinfo.CommonPlayerInfo(player_id=p)
        info = pl.get_normalized_dict()
        info = info.get('CommonPlayerInfo')[0]
        position = [info.get('POSITION')]
        oppTeamID = info.get('TEAM_ID')
        if position[0][0] == playerPosition:
            defenders.append(info.get('PERSON_ID'))
        elif position[0][0] == "F" and playerPosition == "C":
            defenders.append(info.get('PERSON_ID'))

    if len(defenders) < 1:
        for p in opposingTeamLineup:
            time.sleep(.600)
            pl = commonplayerinfo.CommonPlayerInfo(player_id=p)
            info = pl.get_normalized_dict()
            info = info.get('CommonPlayerInfo')[0]
            position = [info.get('POSITION')]
            oppTeamID = info.get('TEAM_ID')
            if playerPosition == "G" and position[0][0] == "F":
                defenders.append(info.get("PERSON_ID"))
                break
            elif playerPosition == "F" and position[0][0] == "G":
                defenders.append(info.get("PERSON_ID"))
                break
            elif playerPosition == "G" and position[0][0] == "F":
                defenders.append(info.get("PERSON_ID"))
                break

    print("DEFENDERS: ", defenders)

    if len(defenders) < 1:
        return "N/A"
    defStats = pd.DataFrame()
    for d in defenders:
        time.sleep(.600)
        defStats = pd.concat([defStats,defensiveStats(d,oppTeamID)])
    
    #defensive stats for 2s and 3s
    d_3s = defStats.loc[defStats["DEFENSE_CATEGORY"] == "3 Pointers"]["D_FG_PCT"].to_numpy()
    d_2s = defStats.loc[defStats["DEFENSE_CATEGORY"]== "2 Pointers"]["D_FG_PCT"].to_numpy()

    print(defStats)

    #defensiveStats(playerId)

    #------ Get average open looks/contested looks/shots at rim ------#
    fga = lastTen['FGA'].to_numpy()
    contestedatt = trackBox['contestedFieldGoalsAttempted'].to_numpy()
    uncontestedatt = trackBox['uncontestedFieldGoalsAttempted'].to_numpy()
    rimatt = trackBox['defendedAtRimFieldGoalsAttempted'].to_numpy()

    pctAtRim =[]
    pctUnContest = []
    pctContest = []

    contestedConf = mean_confidence_interval(contestedatt)
    uncontestedConf = mean_confidence_interval(uncontestedatt)

    
    count = 0
  
    #TODO: Add trend on track details
    #NOTE:  Contested at rim is also under all contested 

    for atts in fga:
        #print("FGA: ", atts, "Contest: ", contestedatt[count], "unc: ",  uncontestedatt[count], "Rim: ", rimatt[count])
        pctContest.append(contestedatt[count]/atts)
        pctUnContest.append(uncontestedatt[count]/atts)
        pctAtRim.append(rimatt[count]/atts)
        count = count + 1
    #average makes on shot types
    contestedpct = trackBox['contestedFieldGoalPercentage'].to_numpy()
    openpct = trackBox['uncontestedFieldGoalsPercentage'].to_numpy()
    rimpct = trackBox['defendedAtRimFieldGoalPercentage'].to_numpy()
    contpctConf = mean_confidence_interval(contestedpct)
    openpctConf = mean_confidence_interval(openpct)
    contestedpct = np.sum(contestedpct)/len(contestedpct)
    rimpct = np.sum(rimpct)/len(rimpct)
    openpct = np.sum(openpct)/len(openpct)

    min = contpctConf[1]* contestedConf[1]
    mid = contpctConf[0]* contestedConf[0]
    top = contpctConf[2]* contestedConf[2]
    contfgConf = []
    contfgConf.append(min)
    contfgConf.append(mid)
    contfgConf.append(top)
    uncontFgConf = []

    min = openpctConf[1] * uncontestedConf[1]
    mid = openpctConf[0] * uncontestedConf[0]
    top = openpctConf[2] * uncontestedConf[2]
    uncontFgConf.append(min)
    uncontFgConf.append(mid)
    uncontFgConf.append(top)

    min = uncontFgConf[0] + contfgConf[0]
    mid = uncontFgConf[1] + contfgConf[1]
    top = uncontFgConf[2] + contfgConf[2]
    fgConf = []
    fgConf.append(min)
    fgConf.append(mid)
    fgConf.append(top)

    contestedJumpers = trackdict["contestedFieldGoalsAttempted"] - trackdict['defendedAtRimFieldGoalsAttempted']

    contestedJumpers = contestedJumpers * contestedpct
    layups = trackdict['defendedAtRimFieldGoalsAttempted'] * rimpct
    openShots = trackdict['uncontestedFieldGoalsAttempted'] * openpct

    #FGM sum By Shot Type Projections 
    sumFGM = contestedJumpers+layups+openShots
    
    # ------------------------------
    #NOTE: Assits by passes per Assist
    # ------------------------------


    passes = np.sum(trackBox['passes'].to_numpy())/len(trackBox['passes'].to_numpy())
    assists = np.sum(trackBox['assists'].to_numpy())/len(trackBox['assists'].to_numpy())

    passPerAssists = assists/passes

    print("passPerAssists", passPerAssists)
    
    astByPass = trackdict['passes'] * passPerAssists

    print("Ast By Pass: ", astByPass, "Ast Proj: ", trackdict['assists'])

    #contestproj = trackdict["contestedFieldGoalsAttempted"]
    #contestedproj = contestedproj[0] * contestedpct

    # ------------------------------
    #NOTE: Rebounds By Reb Chances
    # ------------------------------

    rebChances = trackBox['reboundChancesTotal']
    print("REB CHANCES: ", rebChances)
    rebChancesAvg =  np.sum(trackBox['reboundChancesTotal'].to_numpy())/len(trackBox['reboundChancesTotal'].to_numpy())

    rebChancesProj = trackdict['reboundChancesTotal']
    rebs = lastTen['REB'].to_numpy()
    rebChances = trackBox['reboundChancesTotal'].to_numpy()
    reb_pct = 0
    count = 0
    for reb in rebs:
        #TODO: FIX THIS FOR ZERO DIVISION
        if reb != 0 and rebChances[count] != 0:
            reb_pct = (reb_pct + reb/rebChances[count])
            count = count+ 1
        else: 
            count = count+1

    reb_pct = reb_pct / count

    print ("AVG: ", rebChancesAvg, "PROJ: ", rebChancesProj)

    rebs2 = rebChancesProj * reb_pct

   
    print(lastTenAdvBoxScores['pace'][(lastTenAdvBoxScores["pace"] >= 100) & (lastTenAdvBoxScores["pace"] <= 150)])

    last10pts = lastTen["PTS"].to_numpy()
    last10reb = lastTen["REB"].to_numpy()
    last10ast = lastTen["AST"].to_numpy()

    last10usg = lastTenAdvBoxScores["usagePercentage"].to_numpy()
    last10possesions = lastTenAdvBoxScores["pace"].to_numpy()
    poss = lastTenAdvBoxScores["possessions"].to_numpy()

    ptsPerPos = []
    astPerPos = []
    rebPerPos = []

    possAvg = np.sum(poss)/len(poss)
    count = 0
    vararray = []

    for pts in last10pts:
        usgToPts = pts/last10usg[count]
        vararray.append(usgToPts)
        #get Reb/pos
       # rebPerPos.append(last10reb[count]/last10possesions[count])
        astPerPos.append(last10ast[count]/last10possesions[count])
        ptsPerPos.append(pts/last10possesions[count])
        count = count +1

    print("Pts PerPos: ", ptsPerPos)
    print("AST PerPos: ", astPerPos)
   # print("REB PerPos: ", rebPerPos)
    
    mets = teamestimatedmetrics.TeamEstimatedMetrics()
    dfs = mets.get_data_frames()[0]

    oppTeamDefRtg = dfs.loc[dfs['TEAM_NAME'] == opposingTeam]['E_DEF_RATING']
    tem =dfs.loc[dfs['TEAM_NAME'] == "Indiana Pacers"]

    opposingTeam = tem['TEAM_ID'].values[0]


    teamPace = dfs.loc[dfs['TEAM_NAME'] == team]['E_PACE']

    if teamPace.empty:
        teamPace = 100
    else:
        teamPace = teamPace.values[0]

    dic = matchupsrollup.MatchupsRollup(def_team_id_nullable=opposingTeam, off_player_id_nullable=player[0]['id']).get_data_frames()[0]
    
    if dic.empty:
        print("empty matchup")
        ptsvteam = None
        astvteam = None
        blkvteam = None
    else:
        row = dic.loc[dic["POSITION"] == "TOTAL"]
        gp = row['GP'].values[0]
        ptsvteam = row['PLAYER_PTS'].values[0]/gp
        astvteam = row['MATCHUP_AST'].values[0]/gp
        blkvteam = row["MATCHUP_BLK"].values[0]/gp
        print(ptsvteam, astvteam, blkvteam)

    

    #PLAYERS OFF LOOKUP
    offDic = {}
    count = 0
    for key in playersOut:
        if playersOut[key] == team:
            print("TEAM PLAYER FOUND")
            if not offDic:
                offDic1 = onOff(player[0]['id'], key)
                print(offDic1)
                offDic = offDic1
                count = count +1
            else:
                count = count + 1
                print("TEAM PLAYER FOUND2 ")
                onOffDict = onOff(player[0]['id'], key)
                print(onOffDict)
                if onOffDict:
                    offDic["PTSOFF"] = offDic["PTSOFF"]*(0.5 + 0.5/count) + onOffDict['PTSOFF'] *(0.5/count)
                    offDic["ASTOFF"] = offDic["ASTOFF"]*(0.5 + 0.5/count) + onOffDict['ASTOFF'] *(0.5/count)
                    offDic["FGAOFF"] = offDic["FGAOFF"]*(0.5 + 0.5/count) + onOffDict['FGAOFF'] *(0.5/count)
                    offDic["3sOFF"] = offDic["3sOFF"]*(0.5 + 0.5/count) + onOffDict['3sOFF'] *(0.5/count)
                    offDic["REBOFF"] = offDic["REBOFF"]*(0.5 + 0.5/count) + onOffDict['REBOFF'] *(0.5/count)
                    offDic["STLOFF"] = offDic["STLOFF"]*(0.5 + 0.5/count) + onOffDict['STLOFF'] *(0.5/count)
                    offDic["BLKOFF"] = offDic["BLKOFF"]*(0.5 + 0.5/count) + onOffDict['BLKOFF'] *(0.5/count)

    if not offDic:
        print("EMPTY")
    else:
        print("OFF DICT: ", offDic)
    

    #TEAM DEFENSE

    teamvpts, teamvast, teamvreb = teamdef(team, playerPosition)

    offset = teamPace-100
    ptsper100pos = oppTeamDefRtg/100
    pacenRating = oppTeamDefRtg + (offset*ptsper100pos)

    ptsavg = np.sum(ptsPerPos)/len(ptsPerPos)
    astavg = np.sum(astPerPos)/len(astPerPos)
    #rebavg = np.sum(rebPerPos)/len(rebPerPos)

    usgavg = np.sum(last10usg)/len(last10usg)

    usgToPoints = usgavg* (np.sum(vararray)/len(vararray))

    ptsTotal = ptsavg * pacenRating
    astTotal = astavg * pacenRating

    count = 0
   # for gameid in gameIds:
    #    if count == 0: 
    #        time.sleep(.600)
   #         df = getAdvBoxScore(gameid, player[0]['id'])
    #        totalBoxScore = df
    #    else:
    #        time.sleep(.600)
    #        df = getAdvBoxScore(gameid, player[0]['id'])
    #        totalBoxScore = pandas.concat([totalBoxScore, df])
    #    count = count+1
        

    #print("TOTAL BOX: ", totalBoxScore)



    if gamesdf2022.empty or gamesdf2023.empty:
        return "N/A"
    #print(gamesdf2022)

    matchup = gamesdf2023.iloc[0]['MATCHUP']
    if matchup:
        teams = matchup.split(" ")
    else:
        return "N/A"

    if team != teams[0]:
        team = teams[0]

    columnsOfValue = ['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'REB', 'AST', 'STL', 'BLK', 'PF', 'PTS','FTA','FTM','FG_PCT', 'FG3_PCT','FT_PCT']
    projDict = {}
    for col in columnsOfValue:
        proj = linearProjection(gamesdf2023, col, gamesdf2022)
        projL10 = linearOneProjection(lastTen, col)
        nlProj = nonlinearProjection(gamesdf2023, col)
        projnl = sum(nlProj)/len(nlProj)
        if col == "PTS":
            print("PROJ: ", proj[0])
            print("USG: ", usgToPoints)
            print(ptsTotal)
            if last5ptsavg == None:
                last5ptsavg = projL10[0]
            if type(usgToPoints) ==list:
                usgToPoints = usgToPoints[0]
            if ptsvteam != None:
                proj[0] = proj[0]*0.15 +usgToPoints*0.1+  projL10[0]*0.1 + projnl *0.1 + ptsvteam *0.15+ last5ptsavg *0.2 + ptsperminProj *0.2
            else:
                proj[0] = proj[0]*0.15 +usgToPoints*0.3 + projL10[0]*0.1 + projnl *0.3 + last5ptsavg *0.15

            print(proj[0])
            print("USG2PTS: ", usgToPoints)
            print("PACE&RATING TO PTS: ", ptsTotal )
            print("ADJ RATING: ", pacenRating)
        elif col == "AST":
            if last5astavg == None:
                last5astavg = projL10[0]
            if astvteam != None:
                proj[0] = proj[0]*0.1+ astByPass*0.15 + projL10[0]*0.1 + projnl *0.10 + astvteam*0.15 + last5astavg *0.2 + astperminProj *0.2
            else:
                proj[0] = proj[0]*0.1 + astByPass*0.2 + projL10[0]*0.1 + projnl *0.2+ last5astavg *0.4
        elif col == "FGM":
            proj[0] = proj[0]*0.4 + sumFGM *0.25 + projL10[0] * 0.15  + projnl *0.2
        elif col == "REB":
            if last5rebavg == None:
                last5rebavg = projL10[0]
            proj[0] = proj[0]*0.15 + rebs2 * 0.1 + projL10[0] * 0.15 +  + projnl *0.2+ last5rebavg *0.2 + rebperminProj *0.2
        elif col == "FTM": 
            proj[0]*0.5 + projL10[0] * 0.3 +  + projnl *0.2
        elif col == "FTA": 
            proj[0]*0.5 + projL10[0] * 0.3 + projnl *0.2
        elif col == "STL": 
            proj[0]*0.5 + projL10[0] * 0.3 + projnl *0.2
        elif col == "BLK": 
            proj[0]*0.5 + projL10[0] * 0.3  + projnl *0.2
        elif col == "FG_PCT": 
            proj[0]*0.5 + projL10[0] * 0.3 + projnl *0.2
        elif col == "FG3_PCT": 
            proj[0]*0.5 + projL10[0] * 0.3 + projnl *0.2
        elif col == "FT_PCT": 
            proj[0]*0.5 + projL10[0] * 0.3 + projnl *0.2
        elif col == "MIN": 
            proj[0]*0.5 + projL10[0] * 0.3 + projnl *0.2
        elif col == "FG3A": 
            proj[0]*0.5 + projL10[0] * 0.3 + projnl *0.2
        elif col == "FGA": 
            proj[0]*0.5 + projL10[0] * 0.3 + projnl *0.2
        

        projDict[col] = proj[0] 

    
    #Gather Relevant Projections
    fg_attempted = projDict["FGA"]
    fg_made = projDict["FGM"]
    fg3_made = projDict["FG3M"]
    fg3_attempted = projDict['FG3A']
    fg3_pct = projDict['FG3_PCT']
    fg_pct = projDict['FG_PCT']
    pts_proj = projDict['PTS']
    ftm_proj = projDict['FTM']
    fg2_att = fg_attempted - fg3_attempted
    pct3offset = 0
    pct2offset = 0

    ptsMinusFt = pts_proj - ftm_proj
    fg2_pct = (fg_made - fg3_made)/fg2_att

    if len(d_3s) > 1:
        d_3s = np.sum(d_3s)/len(d_3s)
    if len(d_2s) > 1:
        d_2s = np.sum(d_2s)/ len(d_2s)
    #d_3s, d_2s

    if fg3_pct - d_3s >= 0:
        pct3offset = fg3_pct - (fg3_pct - d_3s)
    elif fg3_pct - d_3s < 0:
        pct3offset = fg3_pct + ((fg3_pct + d_3s)*0.1)
    

    print(fg2_pct, d_2s)
    if fg2_pct - d_2s >= 0:
        pct2offset = fg2_pct - (fg2_pct - d_2s)
    elif fg2_pct - d_2s < 0:
        pct2offset = fg2_pct + ((fg2_pct + d_2s)*0.1)

    print("%3 OFFSET: ", pct3offset, " PROJ %3: ", fg3_pct)
    print("%2 OFFSET: ", pct2offset, " PROJ %2: ", fg2_pct)

    if pct3offset == 0 or pct2offset == 0:
        return "N/A"

    fg3m = fg3_attempted * pct3offset
    fg2m = fg2_att * pct2offset

    points = (fg2m *2) + (fg3m * 3) + ftm_proj

    print("DEF ADDED: ", points, " NON DEF ADDED: ", projDict['PTS'])

    projDict["PTS"] = (projDict["PTS"] * 0.25) + (points * 0.75)

    if not offDic:
        print("NO PLAYERS OUT")
    else:
        projDict["PTS"] =  projDict["PTS"] *0.6 + offDic["PTSOFF"]*0.4
        projDict["AST"] =  projDict["AST"] *0.6 + offDic["ASTOFF"]*0.4
        projDict["FG3M"] =  projDict["FG3M"] *0.6 + offDic["3sOFF"]*0.4
        projDict["REB"] =  projDict["REB"] *0.6 + offDic["REBOFF"]*0.4
        projDict["BLK"] =  projDict["BLK"] *0.6 + offDic["BLKOFF"]*0.4
        projDict["STL"] =  projDict["STL"] *0.6 + offDic["STLOFF"]*0.4

    # projDict["Team"] = team

    projDict["PTS"] =  projDict["PTS"] *0.7 + (projDict["PTS"] * teamvpts) *0.3
    projDict["AST"] =  projDict["AST"] *0.7 + (projDict["AST"] * teamvast)*0.3
    projDict["REB"] =  projDict["REB"] *0.7 + (projDict["REB"] * teamvreb)*0.3

    if hasattr(projDict['PTS'], "__len__"):
        projDict['PTS'] = projDict['PTS'][0]

    print( "LAST 5: ", last5ptsavg, " Proj: ",  projDict["PTS"])
    
    projDict['ptsConfFloor'] = calcConf[0]
    projDict['ptsConfMid'] = calcConf[1]
    projDict['ptsConfTop'] = calcConf[2]

    projDict['astConfFloor'] = calcConfAst[0]
    projDict['astConfMid'] = calcConfAst[1]
    projDict['astConfTop'] = calcConfAst[2]

    projDict['rebConfFloor'] = calcConfReb[0]
    projDict['rebConfMid'] = calcConfReb[1]
    projDict['rebConfTop'] = calcConfReb[2]
    
    return projDict
    

# Retry Wrapper 
def retry(func, retries=3):
    def retry_wrapper(*args, **kwargs):
        attempts = 0
        while attempts < retries:
            try:
                return func(*args, **kwargs)
            except requests.exceptions.RequestException as e:
                print(e)
                time.sleep(30)
                attempts += 1

    return retry_wrapper

# Get Season Schedule Function 

def getSeasonScheduleFrame(seasons,seasonType): 

    # Get date from string
    def getGameDate(matchup):
        return matchup.partition(' at')[0][:10]

    # Get Home team from string
    def getHomeTeam(matchup):
        return matchup.partition(' at')[2]

    # Get Away team from string
    def getAwayTeam(matchup):
        return matchup.partition(' at')[0][10:]

    # Match nickname from schedule to team table to find ID
    def getTeamIDFromNickname(nickname):
        return teamLookup.loc[teamLookup['nickname'] == difflib.get_close_matches(nickname,teamLookup['nickname'],1)[0]].values[0][0] 
    
    @retry
    def getRegularSeasonSchedule(season,teamID,seasonType):
        season = str(season) + "-" + str(season+1)[-2:] # Convert year to season format ie. 2020 -> 2020-21
        teamGames = cumestatsteamgames.CumeStatsTeamGames(league_id = '00',season = season ,
                                                                      season_type_all_star=seasonType,
                                                                      team_id = teamID).get_normalized_json()

        teamGames = pd.DataFrame(json.loads(teamGames)['CumeStatsTeamGames'])
        teamGames['SEASON'] = season
        return teamGames    
    
    # Get team lookup table
    teamLookup = pd.DataFrame(teams.get_teams())
    
    # Get teams schedule for each team for each season
    scheduleFrame = pd.DataFrame()

    for season in seasons:
        for id in teamLookup['id']:
            time.sleep(1)
            scheduleFrame = pandas.concat([scheduleFrame, getRegularSeasonSchedule(season,id,seasonType)])
            
    scheduleFrame['GAME_DATE'] = pd.to_datetime(scheduleFrame['MATCHUP'].map(getGameDate))
    scheduleFrame['HOME_TEAM_NICKNAME'] = scheduleFrame['MATCHUP'].map(getHomeTeam)
    scheduleFrame['HOME_TEAM_ID'] = scheduleFrame['HOME_TEAM_NICKNAME'].map(getTeamIDFromNickname)
    scheduleFrame['AWAY_TEAM_NICKNAME'] = scheduleFrame['MATCHUP'].map(getAwayTeam)
    scheduleFrame['AWAY_TEAM_ID'] = scheduleFrame['AWAY_TEAM_NICKNAME'].map(getTeamIDFromNickname)
    scheduleFrame = scheduleFrame.drop_duplicates() # There's a row for both teams, only need 1
    scheduleFrame = scheduleFrame.reset_index(drop=True)
            
    return scheduleFrame

# Get Single Game aggregation columns
def getSingleGameMetrics(gameID,homeTeamID,awayTeamID,awayTeamNickname,seasonYear,gameDate):

    @retry
    def getGameStats(teamID,gameID,seasonYear):
        gameStats = cumestatsteam.CumeStatsTeam(game_ids=gameID,league_id ="00",
                                               season=seasonYear,season_type_all_star="Regular Season",
                                               team_id = teamID).get_normalized_json()

        gameStats = pd.DataFrame(json.loads(gameStats)['TotalTeamStats'])

        return gameStats

    data = getGameStats(homeTeamID,gameID,seasonYear)
    data.at[1,'NICKNAME'] = awayTeamNickname
    data.at[1,'TEAM_ID'] = awayTeamID
    data.at[1,'OFFENSIVE_EFFICIENCY'] = (data.at[1,'FG'] + data.at[1,'AST'])/(data.at[1,'FGA'] - data.at[1,'OFF_REB'] + data.at[1,'AST'] + data.at[1,'TOTAL_TURNOVERS'])
    data.at[1,'SCORING_MARGIN'] = data.at[1,'PTS'] - data.at[0,'PTS']

    data.at[0,'OFFENSIVE_EFFICIENCY'] = (data.at[0,'FG'] + data.at[0,'AST'])/(data.at[0,'FGA'] - data.at[0,'OFF_REB'] + data.at[0,'AST'] + data.at[0,'TOTAL_TURNOVERS'])
    data.at[0,'SCORING_MARGIN'] = data.at[0,'PTS'] - data.at[1,'PTS']

    data['SEASON'] = seasonYear
    data['GAME_DATE'] = gameDate
    data['GAME_ID'] = gameID

    return data

#get team game logs
def getGameLogs(gameLogs,scheduleFrame):
    
    # Functions to prepare additional columns after gameLogs table loads
    def getHomeAwayFlag(gameDF):
        gameDF['HOME_FLAG'] = np.where((gameDF['W_HOME']==1) | (gameDF['L_HOME']==1),1,0)
        gameDF['AWAY_FLAG'] = np.where((gameDF['W_ROAD']==1) | (gameDF['L_ROAD']==1),1,0) 

    def getTotalWinPctg(gameDF):
        gameDF['TOTAL_GAMES_PLAYED'] = gameDF.groupby(['TEAM_ID','SEASON'])['GAME_DATE'].rank(ascending=True)
        gameDF['TOTAL_WINS'] = gameDF.sort_values(by='GAME_DATE').groupby(['TEAM_ID','SEASON'])['W'].cumsum()
        gameDF['TOTAL_WIN_PCTG'] = gameDF['TOTAL_WINS']/gameDF['TOTAL_GAMES_PLAYED']
        return gameDF.drop(['TOTAL_GAMES_PLAYED','TOTAL_WINS'],axis=1)

    def getHomeWinPctg(gameDF):
        gameDF['HOME_GAMES_PLAYED'] = gameDF.sort_values(by='GAME_DATE').groupby(['TEAM_ID','SEASON'])['HOME_FLAG'].cumsum()
        gameDF['HOME_WINS'] = gameDF.sort_values(by='GAME_DATE').groupby(['TEAM_ID','SEASON'])['W_HOME'].cumsum()
        gameDF['HOME_WIN_PCTG'] = gameDF['HOME_WINS']/gameDF['HOME_GAMES_PLAYED']
        return gameDF.drop(['HOME_GAMES_PLAYED','HOME_WINS'],axis=1)

    def getAwayWinPctg(gameDF):
        gameDF['AWAY_GAMES_PLAYED'] = gameDF.sort_values(by='GAME_DATE').groupby(['TEAM_ID','SEASON'])['AWAY_FLAG'].cumsum()
        gameDF['AWAY_WINS'] = gameDF.sort_values(by='GAME_DATE').groupby(['TEAM_ID','SEASON'])['W_ROAD'].cumsum()
        gameDF['AWAY_WIN_PCTG'] = gameDF['AWAY_WINS']/gameDF['AWAY_GAMES_PLAYED']
        return gameDF.drop(['AWAY_GAMES_PLAYED','AWAY_WINS'],axis=1)

    def getRollingOE(gameDF):
        gameDF['ROLLING_OE'] = gameDF.sort_values(by='GAME_DATE').groupby(['TEAM_ID','SEASON'])['OFFENSIVE_EFFICIENCY'].transform(lambda x: x.rolling(3, 1).mean())

    def getRollingScoringMargin(gameDF):
        gameDF['ROLLING_SCORING_MARGIN'] = gameDF.sort_values(by='GAME_DATE').groupby(['TEAM_ID','SEASON'])['SCORING_MARGIN'].transform(lambda x: x.rolling(3, 1).mean())

    def getRestDays(gameDF):
        gameDF['LAST_GAME_DATE'] = gameDF.sort_values(by='GAME_DATE').groupby(['TEAM_ID','SEASON'])['GAME_DATE'].shift(1)
        gameDF['NUM_REST_DAYS'] = (gameDF['GAME_DATE'] - gameDF['LAST_GAME_DATE'])/np.timedelta64(1,'D') 
        return gameDF.drop('LAST_GAME_DATE',axis=1)
    
    start = time.perf_counter_ns()

    i = int(len(gameLogs)/2) #Can use a previously completed gameLog dataset

    while i<len(scheduleFrame):


        time.sleep(1)
        gameLogs =  pandas.concat([gameLogs, getSingleGameMetrics(scheduleFrame.at[i,'GAME_ID'],scheduleFrame.at[i,'HOME_TEAM_ID'],
                         scheduleFrame.at[i,'AWAY_TEAM_ID'],scheduleFrame.at[i,'AWAY_TEAM_NICKNAME'],
                         scheduleFrame.at[i,'SEASON'],scheduleFrame.at[i,'GAME_DATE'])])
        
        gameLogs = gameLogs.reset_index(drop=True)

        end = time.perf_counter_ns()

        #Output time it took to load x amount of records
        if i%100 == 0:
            mins = ((end-start)/1e9)/60
            print(i,str(mins) + ' minutes')

        i+=1
        
    # Get Table Level Aggregation Columns
    getHomeAwayFlag(gameLogs)
    gameLogs = getHomeWinPctg(gameLogs)
    gameLogs = getAwayWinPctg(gameLogs)
    gameLogs = getTotalWinPctg(gameLogs)
    getRollingScoringMargin(gameLogs)
    getRollingOE(gameLogs)
    gameLogs = getRestDays(gameLogs)

    return gameLogs.reset_index(drop=True)


def teamWinLoss():

    #Get ScheduleFrame

    seasons = [2022,2023]
    seasonType = 'Regular Season'

    start = time.perf_counter_ns() # Track cell's runtime
    scheduleFrame = getSeasonScheduleFrame(seasons,seasonType)
    end = time.perf_counter_ns()

    secs = (end-start)/1e9
    mins = secs/60
    print(mins)

    gameLogs = pd.DataFrame()
    gameLogs = getGameLogs(gameLogs,scheduleFrame)
    #Create the gameLogs DataFrame
    gameLogs.to_csv('gameLogs.csv')
    def getGameLogFeatureSet(gameDF):
    
        def shiftGameLogRecords(gameDF):
            gameDF['LAST_GAME_OE'] = gameLogs.sort_values('GAME_DATE').groupby(['TEAM_ID','SEASON'])['OFFENSIVE_EFFICIENCY'].shift(1)
            gameDF['LAST_GAME_HOME_WIN_PCTG'] = gameDF.sort_values('GAME_DATE').groupby(['TEAM_ID','SEASON'])['HOME_WIN_PCTG'].shift(1)
            gameDF['LAST_GAME_AWAY_WIN_PCTG'] = gameDF.sort_values('GAME_DATE').groupby(['TEAM_ID','SEASON'])['AWAY_WIN_PCTG'].shift(1)
            gameDF['LAST_GAME_TOTAL_WIN_PCTG'] = gameDF.sort_values('GAME_DATE').groupby(['TEAM_ID','SEASON'])['TOTAL_WIN_PCTG'].shift(1)
            gameDF['LAST_GAME_ROLLING_SCORING_MARGIN'] = gameDF.sort_values('GAME_DATE').groupby(['TEAM_ID','SEASON'])['ROLLING_SCORING_MARGIN'].shift(1)
            gameDF['LAST_GAME_ROLLING_OE'] = gameDF.sort_values('GAME_DATE').groupby(['TEAM_ID','SEASON'])['ROLLING_OE'].shift(1)
        
        
        def getHomeTeamFrame(gameDF):
            homeTeamFrame = gameDF[gameDF['CITY'] != 'OPPONENTS']
            homeTeamFrame = homeTeamFrame[['LAST_GAME_OE','LAST_GAME_HOME_WIN_PCTG','NUM_REST_DAYS','LAST_GAME_AWAY_WIN_PCTG','LAST_GAME_TOTAL_WIN_PCTG','LAST_GAME_ROLLING_SCORING_MARGIN','LAST_GAME_ROLLING_OE','W','TEAM_ID','GAME_ID','SEASON']]

            colRenameDict = {}
            for col in homeTeamFrame.columns:
                if (col != 'GAME_ID') & (col != 'SEASON') :
                    colRenameDict[col] = 'HOME_' + col 

            homeTeamFrame.rename(columns=colRenameDict,inplace=True)

            return homeTeamFrame

        def getAwayTeamFrame(gameDF):
            awayTeamFrame = gameDF[gameDF['CITY'] == 'OPPONENTS']
            awayTeamFrame = awayTeamFrame[['LAST_GAME_OE','LAST_GAME_HOME_WIN_PCTG','NUM_REST_DAYS','LAST_GAME_AWAY_WIN_PCTG','LAST_GAME_TOTAL_WIN_PCTG','LAST_GAME_ROLLING_SCORING_MARGIN','LAST_GAME_ROLLING_OE','TEAM_ID','GAME_ID','SEASON']]

            colRenameDict = {}
            for col in awayTeamFrame.columns:
                if (col != 'GAME_ID') & (col != 'SEASON'):
                    colRenameDict[col] = 'AWAY_' + col 

            awayTeamFrame.rename(columns=colRenameDict,inplace=True)

            return awayTeamFrame
        
        shiftGameLogRecords(gameLogs)
        awayTeamFrame = getAwayTeamFrame(gameLogs)
        homeTeamFrame = getHomeTeamFrame(gameLogs)
        
        return pd.merge(homeTeamFrame, awayTeamFrame, how="inner", on=[ "GAME_ID","SEASON"]).drop(['GAME_ID','AWAY_TEAM_ID','HOME_TEAM_ID'],axis=1)

   # gameLogs.to_csv('gameLogs.csv')
    modelData = getGameLogFeatureSet(gameLogs)
    modelData.to_csv('nbaHomeWinLossModelDataset.csv')


def modelTeamWinLoss(awayTeamID, homeTeamID):

    #Example Output of Game Logs
    gameLogs = pd.read_csv('gameLogs.csv')
    awayData = gameLogs[(gameLogs['TEAM_ID'] == awayTeamID ) & (gameLogs['SEASON'] == '2022-23')].sort_values('GAME_DATE')
    homeData = gameLogs[(gameLogs['TEAM_ID'] == homeTeamID ) & (gameLogs['SEASON'] == '2022-23')].sort_values('GAME_DATE')

    print(awayData, homeData)

    data = pd.read_csv('nbaHomeWinLossModelDataset.csv').drop(['Unnamed: 0'],axis=1)
    data = data.dropna()

    #homeData = homeData.dropna()
    #data = homeData
    data.head(10)
    validation = data[data['SEASON'] == '2022-23']
    modelData = data[data['SEASON'] != '2022-23'].sample(frac=1)
    X = modelData.drop(['HOME_W','SEASON'],axis=1)
    y = modelData['HOME_W']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.33)

    # Standard Scaling Prediction Variables
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    scaled_data_train = scaler.transform(X_train)

    scaler.fit(X_test)
    scaled_data_test = scaler.transform(X_test) 

    model = LogisticRegression()
    model.fit(scaled_data_train,y_train)
    model.score(scaled_data_test,y_test)
    F1Score = cross_val_score(model,scaled_data_test,y_test,cv=12,scoring='f1_macro')
    print("Logistic Model F1 Accuracy: %0.2f (+/- %0.2f)"%(F1Score.mean(), F1Score.std() *2))

    y_pred = model.predict(scaled_data_test)
    print(classification_report(y_test,y_pred))

    #Validation Set review

    # Standard Scaling Prediction Variables
    scaler = preprocessing.StandardScaler()
    scaler.fit(validation.drop(['HOME_W','SEASON'],axis=1))
    scaled_val_data = scaler.transform(validation.drop(['HOME_W','SEASON'],axis=1))

    # How the model performs on unseen data
    y_pred = model.predict(scaled_val_data)
    print(classification_report(validation['HOME_W'],y_pred))

@retry
def findplayer(playerName):
    player = players.find_players_by_full_name(playerName)
    return player

def modelHead2Head(offPlayer, defPlayer):

    fullMatchups = pandas.DataFrame()

    for player in defPlayer:
        matchupVal = matchupsrollup.MatchupsRollup(off_player_id_nullable=offPlayer, def_player_id_nullable=player)
        matchupDf = matchupVal.get_data_frames()[0]
        if matchupDf.empty:
            print("empty DF")
        else:
            fullMatchups = pandas.concat([fullMatchups, matchupDf])

        print(matchupDf)
   #print(matchupDf)

   #pvp = playervsplayer.PlayerVsPlayer(player_id=offPlayer, vs_player_id=defPlayer, season=2022, season_type_playoffs="Regular Season").get_data_frames()

    print(fullMatchups)
    if fullMatchups.empty:
        print("Empty full matchups")
    else:
        fullMatchups.to_csv('h2h.csv', sep='\t', encoding='utf-8')
   
#RUN TO LOAD IN LINEUPS TO JSON PREMODEL
def getJsonLineups():
    lineups, out = readInLineups()

    #TODO: Set full game dictionary?

    allplayers = lineups[[0]].to_numpy()

    playersout = out[[0]].to_numpy()

    playersoutDict ={}

    startingLineupsDict = {}
    startingLineupsByName = {}

    for p in playersout:
        
        time.sleep(.600)
        player = findplayer(p[0])
        print(player)
        if player:
            gameLog2023 = playergamelog.PlayerGameLog(player[0]['id'],2023,'Regular Season')
            gamesdf2023 = gameLog2023.get_data_frames()[0]
            #print(gamesdf2023)
            if gamesdf2023.empty:
                print(p)
            else:
                matchup = gamesdf2023.iloc[0]['MATCHUP']
                teams = matchup.split(" ")
                player_team = teams[0]
                playersoutDict[player[0]['id']] = player_team

        #Get player current team

        #startingLineupsByName[player[0]["full_name"]] = player_team
            with open('playersOutDict.json', 'w', encoding='utf-8') as f:
                json.dump(playersoutDict, f, ensure_ascii=False, indent=4)

   # print(playersoutDict)
    for p in allplayers:
        
        time.sleep(.600)
        player = findplayer(p[0])
        print(player)
        if player:
            gameLog2023 = playergamelog.PlayerGameLog(player[0]['id'],2023,'Regular Season')
            gamesdf2023 = gameLog2023.get_data_frames()[0]
            teams = matchup.split(" ")
            player_team = teams[0]
            startingLineupsDict[player[0]['id']] = player_team
            startingLineupsByName[player[0]["full_name"]] = player_team
        else:
            print("none")
        #print(gamesdf2023)
        if gamesdf2023.empty:
            print(p)
        else:
            matchup = gamesdf2023.iloc[0]['MATCHUP']

        #Get player current team
        
    
    with open('startingLineupsDict.json', 'w', encoding='utf-8') as f:
        json.dump(startingLineupsDict, f, ensure_ascii=False, indent=4)
    with open('startingLineupsByName.json', 'w', encoding='utf-8') as f:
        json.dump(startingLineupsByName, f, ensure_ascii=False, indent=4)


def teamdef(teamName, position):

    from statistics import median

    with open('defensevposition.json') as json_file:
            defensestats = json.load(json_file)

    teamName = "OKC"
    positions = {
        "G": "PG",
        "SG": "SG",
        "SF": "SF",
        "F": "SF",
        "C": "C"
    }

    pp = ""
    for key in positions:
        if key == position:
            pp = positions[key]

    ppgavg = []
    astavg = []
    rebavg = []


    for key in defensestats:

        team = key['TEAM'].split(" ")[0]
        position = key['POSITION'].split(" ")[0]

        if pp == position:
            ppgavg.append( float(key['PTS'].split(" ")[0]))
            astavg.append( float(key['AST'].split(" ")[0]))
            rebavg.append(float(key['REB'].split(" ")[0]))

    leagueAvgPts = median(ppgavg)
    leagueAvgAst = median(astavg)
    leagueAvgReb = median(rebavg)

    print(leagueAvgPts, leagueAvgAst, leagueAvgReb)

    for key in defensestats:
        team = key['TEAM'].split(" ")[0]
        position = key['POSITION'].split(" ")[0]

        if pp == position and team == teamName:
            
            ppg = float(key['PTS'].split(" ")[0])
            ast = float(key['AST'].split(" ")[0])
            reb = float(key['REB'].split(" ")[0])

            ptsRatio = ppg/leagueAvgPts
            astRatio = ast/leagueAvgAst
            rebRatio = reb/leagueAvgReb

            print(ptsRatio, astRatio, rebRatio)
            print("FOUND")
    
    return ptsRatio, astRatio, rebRatio



#### MAIN FUNCTION: Model Games ####
def modelGame():
    getLineups()
    getJsonLineups()

    games = getTodaysGames()
    
    game_dictionary = {}
    ## Model 1 game for now ## TODO: Model All Games on Slate 
    all_games = []

    with open('startingLineupsByName.json') as json_file:
        startingLineupsByName = json.load(json_file)

    with open('startingLineupsDict.json') as json_file:
        startingLineupsDict = json.load(json_file)

    with open('playersOutDict.json') as json_file:
        playersoutDict = json.load(json_file)


    count = 0
    for game in games:
        #print(game)
        homeTeamAbr = game['homeTeam']['teamTricode']
        
        awayTeamAbr = game['awayTeam']['teamTricode']

        homeTeamLineup = []
        awayTeamLineup = []
        homeTeamLineupNames = []
        awayTeamLineupNames = []

        all_games.append(homeTeamAbr + " " + awayTeamAbr)
        for key in startingLineupsDict:

          #  print(startingLineupsDict[key] )

            if startingLineupsDict[key] == homeTeamAbr:
                homeTeamLineup.append(key)
            elif startingLineupsDict[key] == awayTeamAbr:
                awayTeamLineup.append(key)

        print(homeTeamLineup)
        print(awayTeamLineup)

        teams = {
            "ATL": "Atlanta Hawks",
            "BOS": "Boston Celtics",
            "CHA": "Charlotte Hornets",
            "DAL": "Dallas Maverics",
            "CHI": "Chicago Bulls",
            "CLE": "Cleveland Cavaliers",
            "DEN":  "Denver Nuggets",
            "DET":	"Detroit Pistons",
            "GSW":	"Golden State Warriors",
            "HOU": "Houston Rockets",
            "IND" : "Indiana Pacers",
            "LAC" :"Los Angeles Clippers",
            "LAL":"Los Angeles Lakers",
            "MEM":"Memphis Grizzlies",      
            "MIA":"Miami Heat",
            "MIL":"Milwaukee Bucks",
            "MIN":"Minnesota Timberwolves",
            "NOP":"New Orleans Pelicans",
            "NYK":"New York Knicks",
            "BKN":"Brooklyn Nets",
            "OKC":"Oklahoma City Thunder",
            "ORL":"Orlando Magic",
            "PHI":"Philadelphia 76ers",
            "PHX":"Phoenix Suns",
            "POR":"Portland Trail Blazers",
            "SAC":"Sacramento Kings",
            "SAS": "San Antonio Spurs",
            "TOR":"Toronto Raptors",
            "UTA":"Utah Jazz",
            "WAS":"Washington Wizards",
        }

        for key in startingLineupsByName:
            if startingLineupsByName[key] == homeTeamAbr:
                homeTeamLineupNames.append(key)
            elif startingLineupsByName[key] == awayTeamAbr:
                awayTeamLineupNames.append(key)
        
        for player in homeTeamLineupNames:
            time.sleep(.600)
            player_dict = getPlayerProjection(player, teams[homeTeamAbr], teams[awayTeamAbr], awayTeamLineup, playersoutDict)
            game_dictionary[player] = player_dict
    
        for player in awayTeamLineupNames:
            time.sleep(.600)
            player_dict = getPlayerProjection(player, teams[homeTeamAbr], teams[awayTeamAbr], homeTeamLineup, playersoutDict)
            game_dictionary[player] = player_dict

        #modelHead2Head(homeTeamLineup[0], awayTeamLineup)

       
        #print(homeTeamAbr, awayTeamAbr)
       # home = teams.find_team_by_abbreviation(homeTeamAbr)
       # away = teams.find_team_by_abbreviation(awayTeamAbr)
    #modelTeamWinLoss(away['id'], home['id'])

  

    #print(startingLineupsDict)


        #homeTeamLineupdf = lineups.loc[lineups[1] == homeTeamAbr]
       # awayTeamLineupdf = lineups.loc[lineups[1] == awayTeamAbr]

        #print(homeTeamLineupdf)

    #issue with PG team on lineups
   # print(awayTeamLineupdf)

        #teamEstimates = teamestimatedmetrics.TeamEstimatedMetrics(00, 2023, 'Regular Season')

    #get arrays of each team lineups
       # homeTeamLineup = homeTeamLineupdf[0].to_numpy()
       # awayTeamLineup = awayTeamLineupdf[0].to_numpy()

  #  offPlayerName = homeTeamLineupdf.loc[homeTeamLineupdf[2] == "SF"]
  #  defPlayerName = awayTeamLineupdf.loc[awayTeamLineupdf[2]=="SF"]
  #  print(offPlayerName)
   # offPlayerName = offPlayerName.iloc[0][0]
   # defPlayerName = defPlayerName.iloc[0][0]
    
   # offPlayer = players.find_players_by_full_name(offPlayerName)
   # defPlayer = players.find_players_by_full_name(defPlayerName)
    
    #print(winprobabilitypbp.WinProbabilityPBP(games[0]['gameId'], RunType.default).get_data_frames())
    
    #print(offPlayerName, defPlayerName)

    #modelHead2Head(offPlayer[0]['id'], defPlayer[0]['id'])
   

    #run projections and save to game dictionary
    
    #for player in awayTeamLineupNames:
   #    player_dict = getPlayerProjection(player, homeTeamAbr)
    #    game_dictionary[player] = player_dict
    
    print(game_dictionary)
    with open('projections.json', 'w', encoding='utf-8') as f:
        json.dump(game_dictionary, f, ensure_ascii=False, indent=4)
    
#-------------------#
#Gets data for spreadsheets
#-------------------#

#teamWinLoss()

#modelTeamWinLoss()


#-------------------#
#models game stats
#-------------------#

   
#getPlayerProjection("Damian Lillard", "Milwaukee Bucks", "Indiana Pacers")
#gameLog2023 = playergamelog.PlayerGameLog("203507",2023,'Regular Season').get_data_frames()[0]
#nl = nonlinearProjection(gameLog2023, "PTS")
#print(sum(nl)/len(nl))

modelGame()

#matchupVal = matchupsrollup.MatchupsRollup(off_player_id_nullable=1628368, def_player_id_nullable=1626156)
#matchupDf = matchupVal.get_data_frames()[0]

#print(matchupDf['MATCHUP_MIN'])
#print(matchupDf['MATCHUP_FG_PCT'])

#dic = playervsplayer.PlayerVsPlayer(vs_player_id=1626156, player_id=1628368).get_data_frames()
#print(dic)


#tests = playerdashptshotdefend.PlayerDashPtShotDefend(player_id='2544', team_id='0').get_data_frames()[0]
#test = playerdashptshotdefend.PlayerDashPtShotDefend(player_id='203552',team_id='0').get_data_frames()[0]

#test = pd.concat([test, tests])
#print(test)
#d_3s = test.loc[test["DEFENSE_CATEGORY"] == "3 Pointers"]["D_FG_PCT"].to_numpy()
#d_2s = test.loc[test["DEFENSE_CATEGORY"]== "2 Pointers"]["D_FG_PCT"].to_numpy()
#print(d_2s)
#if not test:
#    raise Exception('fail', 'playerdashptshotdefend')
#test = playerdashptshotdefend.PlayerDashPtShotDefend(player_id='203552',
  #      team_id='0').get_data_frames()
#print(test)
#sys.sleep()
#gameLog2023 = playergamelog.PlayerGameLog(203552,2023,'Regular Season').get_data_frames()[0]
#print(gameLog2023)
#games = gameLog2023['Game_ID'].to_numpy()
#d = playerdashptshotdefend.PlayerDashPtShotDefend(player_id="203552", team_id="1610612744", date_to_nullable="12/08/2023", date_from_nullable="10/25/2023")

#print(d.get_data_frames())

### TEST NN ####

"""
player = players.find_players_by_full_name("Domantas Sabonis")
gameLog2023 = playergamelog.PlayerGameLog(player[0]['id'],2023,'Regular Season')
gamesdf2023 = gameLog2023.get_data_frames()[0]

gameLog2022 = playergamelog.PlayerGameLog(player[0]['id'],2022,'Regular Season')
gamesdf2022 = gameLog2022.get_data_frames()[0]

pts2023 = gamesdf2023['PTS'].tolist()
pts2022 = gamesdf2022['PTS'].tolist()

length = len(pts2022)
#outPutSet = array()
inputSets = []
count = 0
tempInSet = []
outputSet = []
overallCount = 0
#print(pts2022, pts2023)
for pt in pts2022:
    if overallCount + 3 <= length:
        if count < 3:
            tempInSet.append(pt)
            count = count+1
            overallCount = overallCount +1
        else:
            outputSet.append(pt)
            inputSets.append(tempInSet)
            tempInSet = []
            count = 0
            overallCount = overallCount +1
    else:
        break

print("Inputs: ", inputSets, "outs: ", outputSet)
count = 0
tempInSet = []
length = len(pts2023)
for pt in pts2023:
    if overallCount + 3 <= length-3:
        if count < 3:
            tempInSet.append((pt/1000))
            count = count+1
            overallCount = overallCount +1
        else:
            outputSet.append(pt)
            inputSets.append(tempInSet)
            tempInSet = []
            count = 0
            overallCount = overallCount +1
    else:
        break

setToPredict = [pts2023.pop(), pts2023.pop(1), pts2023.pop(2)]


print("Inputs: ", inputSets, "outs: ", outputSet, "Predict: ", setToPredict)
runNNPrediction(inputSets, outputSet, setToPredict)
"""











