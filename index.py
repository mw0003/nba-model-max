# NBA PLAYER PROJECTION MODEL

# API https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/static/players.md

#career stats
from queue import Empty
from nba_api.stats.endpoints import playercareerstats, playergamelog, teamestimatedmetrics
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
from nba_api.stats.endpoints import cumestatsteamgames, cumestatsteam, gamerotation, matchupsrollup,winprobabilitypbp,boxscoreadvancedv3, playervsplayer
import pandas as pd
import numpy as np
import json
import difflib
import time
import requests

from nn import *

from runtype import Dispatch
dp = Dispatch()

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
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing 
from sklearn.metrics import classification_report

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

#get adv box score by player
def getAdvBoxScore(gameid, playerid):
    advBoxScore = boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=gameid).player_stats
    advBoxScoreDf = advBoxScore.get_data_frame()

    playerDFInstance = advBoxScoreDf.loc[advBoxScoreDf["personId"] == playerid]
    newDF = playerDFInstance[['minutes', 'assistPercentage', 'assistRatio','reboundPercentage','effectiveFieldGoalPercentage','trueShootingPercentage','usagePercentage','pace','possessions']].copy()
   # print(newDF)
    return newDF

#run player projections
def getPlayerProjection(playerName, team):
    
    player = players.find_players_by_full_name(playerName)
    print(player[0])

    gameLog2023 = playergamelog.PlayerGameLog(player[0]['id'],2023,'Regular Season')
    gameLog2022 = playergamelog.PlayerGameLog(player[0]['id'],2022,'Regular Season')
    
    #convert playergamelog objects to DFs
    gamesdf2023 = gameLog2023.get_data_frames()[0]
    gamesdf2022 = gameLog2022.get_data_frames()[0]

    #Get last 10 games
    lastTen = gamesdf2023.head(10)

    #gameIds = gamesdf2023["Game_ID"].to_list()
    #print(lastTen)

    gameIds = lastTen["Game_ID"].to_numpy()
    playerID = lastTen.iloc[0]["Player_ID"]
    lastTenAdvBoxScores = pd.DataFrame()
    for id in gameIds:
        df = getAdvBoxScore(id, playerID)
        lastTenAdvBoxScores = pd.concat([lastTenAdvBoxScores, df])

    print(lastTenAdvBoxScores)
    print(lastTen)

    last10pts = lastTen["PTS"].to_numpy()
    last10usg = lastTenAdvBoxScores["usagePercentage"].to_numpy()

    count = 0
    vararray = []
    for pts in last10pts:
        usgToPts = pts/last10usg[count]
        vararray.append(usgToPts)

    ptsavg = np.sum(last10pts)/10
    usgavg = np.sum(last10usg)/10
    print(usgavg)
    print(vararray)

    print("From SUM: ", usgavg* (np.sum(vararray)/10))
    print("From Arrays: ", ptsavg)

    sys.exit()
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

    teams = matchup.split(" ")

    if team != teams[0]:
        team = teams[0]

    columnsOfValue = ['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'REB', 'AST', 'STL', 'BLK', 'PF', 'PTS','FTA','FTM','FG_PCT', 'FG3_PCT','FT_PCT']
    projDict = {}
    for col in columnsOfValue:
        proj = linearProjection(gamesdf2023, col, gamesdf2022)
        projDict[col] = proj[0]

    # projDict["Team"] = team
    

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
   

#### MAIN FUNCTION: Model Games ####
def modelGame():
    #getLineups()
    games = getTodaysGames()
    lineups, out = readInLineups()

    #TODO: Set full game dictionary?

    game_dictionary = {}
    allplayers = lineups[[0]].to_numpy()
    startingLineupsDict = {}

    startingLineupsByName = {}

    for p in allplayers:
        
        time.sleep(.600)
        player = findplayer(p[0])
        print(player)
        gameLog2023 = playergamelog.PlayerGameLog(player[0]['id'],2023,'Regular Season')
        gamesdf2023 = gameLog2023.get_data_frames()[0]
        #print(gamesdf2023)
        if gamesdf2023.empty:
            print(p)
        else:
            matchup = gamesdf2023.iloc[0]['MATCHUP']

        #Get player current team
        teams = matchup.split(" ")
        player_team = teams[0]
        startingLineupsDict[player[0]['id']] = player_team
        startingLineupsByName[player[0]["full_name"]] = player_team
    


    ## Model 1 game for now ## TODO: Model All Games on Slate 
    all_games = []
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

        for key in startingLineupsByName:
            if startingLineupsByName[key] == homeTeamAbr:
                homeTeamLineupNames.append(key)
            elif startingLineupsByName[key] == awayTeamAbr:
                awayTeamLineupNames.append(key)
        
        for player in homeTeamLineupNames:
            time.sleep(.600)
            player_dict = getPlayerProjection(player, homeTeamAbr)
            game_dictionary[player] = player_dict
    
        for player in awayTeamLineupNames:
            time.sleep(.600)
            player_dict = getPlayerProjection(player, awayTeamAbr)
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

getPlayerProjection("Luka Doncic", "Dallas Mavericks")



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











