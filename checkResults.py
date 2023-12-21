from datetime import datetime, timedelta
import json
import time
from turtle import home


from nba_api.stats.endpoints import boxscoretraditionalv3
from nba_api.live.nba.endpoints import boxscore
from nba_api.live.nba.endpoints import scoreboard

#Get yesterdays date
def yesterday(frmt='%Y-%m-%d', string=True):
    yesterday = datetime.now() - timedelta(1)
    if string:
        return yesterday.strftime(frmt)
    return yesterday

with open('lines.json') as json_file:
    lines = json.load(json_file)
 
with open('data.json') as json_file:
    proj = json.load(json_file)
 

#games = ['0022300295', '0022300294','0022300293','0022300292','0022300297','0022300296','0022300298']


correctPicks = 0
incorrectPicks= 0

correctPts = 0
incorrectPts = 0

correctAst = 0
incorrectAst = 0

correctReb = 0
incorrectReb = 0

correctStl = 0
incorrectStl = 0

correctBlk = 0 
incorrectBlk = 0

correctThree = 0
incorrectThree = 0

results ={}

#UNCOMMENT IF RUNNING BEFORE 10AM NEXT DAY
board = scoreboard.ScoreBoard()
games = board.games.get_dict()

#print(games)

for game in games:

    time.sleep(.600)
    
    #print(box)
    #print("GAME: ", game)

    ##IF Running before 10Am MTN next day USE BOXSCORE
    
    box = boxscore.BoxScore(game_id=game['gameId'])


    awayStats = box.away_team_player_stats.get_dict()
    homeStats = box.home_team_player_stats.get_dict()

    ##ELSE USE BOXSCORE TRADITIONAL

   # box = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game).get_dict()
    
    #homeStats = box['boxScoreTraditional']['homeTeam']['players']
    #awayStats = box['boxScoreTraditional']['awayTeam']['players']
    #print(box['player_stats'])

   #print(homeStats)

    for pl in homeStats:
        #print(pl)
        for item in proj:

            if (pl['firstName'] + " " +pl['familyName'] ==item):

                print(item)
                #check for NA projections
                if proj[item] != "N/A":
                    #handle Points
                    if pl['statistics']['points'] > -1 and 'fanduel_pts' in proj[item]:
                        if pl['statistics']['points'] > proj[item]['fanduel_pts'] and proj[item]['pick_pts'] == "OVER":
                            print("CORRECT")
                            results[item] = proj[item]
                            results[item]['pts_res'] = "W"
                            correctPicks = correctPicks +1
                            correctPts = correctPts +1
                        elif pl['statistics']['points'] < proj[item]['fanduel_pts'] and proj[item]['pick_pts'] == "UNDER":
                            print("CORRECT")
                            results[item] = proj[item]
                            results[item]['pts_res'] = "W"
                            correctPicks = correctPicks +1
                            correctPts = correctPts +1
                        else:
                            print("INCORRECT")
                            results[item] = proj[item]
                            results[item]['pts_res'] = "L"
                            incorrectPicks = incorrectPicks + 1
                            incorrectPts = incorrectPts +1

                    if pl['statistics']['assists'] > -1 and 'fanduel_ast' in proj[item]:

                        if pl['statistics']['assists'] > proj[item]['fanduel_ast'] and proj[item]['pick_ast'] == "OVER":
                            print("CORRECT")
                            results[item] = proj[item]
                            results[item]['ast_res'] = "W"
                            correctPicks = correctPicks +1
                            correctAst = correctAst +1
                        elif pl['statistics']['assists'] < proj[item]['fanduel_ast'] and proj[item]['pick_ast'] == "UNDER":
                            print("CORRECT")
                            results[item] = proj[item]
                            results[item]['ast_res'] = "W"
                            correctPicks = correctPicks +1
                            correctAst = correctAst +1
                        else:
                            print("INCORRECT")
                            results[item] = proj[item]
                            results[item]['ast_res'] = "L"
                            incorrectPicks = incorrectPicks + 1
                            incorrectAst = incorrectAst +1

                    if pl['statistics']['reboundsTotal'] > -1 and 'fanduel_reb' in proj[item]:

                        if pl['statistics']['reboundsTotal'] > proj[item]['fanduel_reb'] and proj[item]['pick_reb'] == "OVER":
                            print("CORRECT")
                            results[item] = proj[item]
                            results[item]['reb_res'] = "W"
                            correctPicks = correctPicks +1
                            correctReb = correctReb +1
                        elif pl['statistics']['reboundsTotal'] < proj[item]['fanduel_reb'] and proj[item]['pick_reb'] == "UNDER":
                            print("CORRECT")
                            results[item] = proj[item]
                            results[item]['reb_res'] = "W"
                            correctPicks = correctPicks +1
                            correctReb = correctReb +1
                        else:
                            print("INCORRECT")
                            results[item] = proj[item]
                            results[item]['reb_res'] = "L"
                            incorrectPicks = incorrectPicks + 1
                            incorrectReb = incorrectReb +1

                    if pl['statistics']['steals'] > -1 and 'fanduel_stl' in proj[item]:

                        if pl['statistics']['steals'] > proj[item]['fanduel_stl'] and proj[item]['pick_stl'] == "OVER":
                            print("CORRECT")
                            results[item] = proj[item]
                            results[item]['stl_res'] = "W"
                            correctPicks = correctPicks +1
                            correctStl = correctStl +1
                        elif pl['statistics']['steals'] < proj[item]['fanduel_stl'] and proj[item]['pick_stl'] == "UNDER":
                            print("CORRECT")
                            results[item] = proj[item]
                            results[item]['stl_res'] = "W"
                            correctPicks = correctPicks +1
                            correctStl = correctStl +1
                        else:
                            print("INCORRECT")
                            results[item] = proj[item]
                            results[item]['stl_res'] = "L"
                            incorrectPicks = incorrectPicks + 1
                            incorrectStl = incorrectStl +1

                    if pl['statistics']['threePointersMade'] > -1 and 'fanduel_threes' in proj[item]:

                        if pl['statistics']['threePointersMade'] > proj[item]['fanduel_threes'] and proj[item]['pick_3s'] == "OVER":
                            print("CORRECT")
                            results[item] = proj[item]
                            results[item]['threes_res'] = "W"
                            correctPicks = correctPicks +1
                            correctThree = correctThree +1
                        elif pl['statistics']['threePointersMade'] < proj[item]['fanduel_threes'] and proj[item]['pick_3s'] == "UNDER":
                            print("CORRECT")
                            results[item] = proj[item]
                            results[item]['threes_res'] = "W"
                            correctPicks = correctPicks +1
                            correctThree = correctThree +1
                        else:
                            print("INCORRECT")
                            results[item] = proj[item]
                            results[item]['threes_res'] = "L"
                            incorrectPicks = incorrectPicks + 1
                            incorrectThree = incorrectThree +1

                    if pl['statistics']['blocks'] > -1 and 'fanduel_blk' in proj[item]:

                        if pl['statistics']['blocks'] > proj[item]['fanduel_blk'] and proj[item]['pick_3s'] == "OVER":
                            print("CORRECT")
                            results[item] = proj[item]
                            results[item]['threes_res'] = "W"
                         #   correctPicks = correctPicks +1
                         #   correctBlk = correctBlk +1
                        elif pl['statistics']['blocks'] < proj[item]['fanduel_blk'] and proj[item]['pick_blk'] == "UNDER":
                            print("CORRECT")
                            results[item] = proj[item]
                            results[item]['blk_res'] = "W"
                          #  correctPicks = correctPicks +1
                          #  correctBlk = correctBlk +1
                        else:
                            print("INCORRECT")
                            results[item] = proj[item]
                            results[item]['blk_res'] = "L"
                            #incorrectPicks = incorrectPicks + 1
                           # incorrectBlk = incorrectBlk +1
               # print(proj[item])

           # results[line] = line
            #if row['points'] > line['']
           # results[line]['pts_res'] = "W"
            


for pl in awayStats:
        #print(pl)
        for item in proj:
            if (pl['firstName'] + " " +pl['familyName'] ==item):
                print("AWAY TEAM")
                #check for NA projections
                if proj[item] != "N/A":
                    #handle Points
                    if pl['statistics']['points'] > -1 and 'fanduel_pts' in proj[item]:
                        if pl['statistics']['points'] > proj[item]['fanduel_pts'] and proj[item]['pick_pts'] == "OVER":
                            print("CORRECT")
                            results[item] = proj[item]
                            results[item]['pts_res'] = "W"
                            correctPicks = correctPicks +1
                            correctPts = correctPts +1
                        elif pl['statistics']['points'] < proj[item]['fanduel_pts'] and proj[item]['pick_pts'] == "UNDER":
                            print("CORRECT")
                            results[item] = proj[item]
                            results[item]['pts_res'] = "W"
                            correctPicks = correctPicks +1
                            correctPts = correctPts +1
                        else:
                            print("INCORRECT")
                            results[item] = proj[item]
                            results[item]['pts_res'] = "L"
                            incorrectPicks = incorrectPicks + 1
                            incorrectPts = incorrectPts + 1

                    if pl['statistics']['assists'] > -1 and 'fanduel_ast' in proj[item]:

                        if pl['statistics']['assists'] > proj[item]['fanduel_ast'] and proj[item]['pick_ast'] == "OVER":
                            print("CORRECT")
                            results[item] = proj[item]
                            results[item]['ast_res'] = "W"
                            correctPicks = correctPicks +1
                            correctAst = correctAst +1
                        elif pl['statistics']['assists'] < proj[item]['fanduel_ast'] and proj[item]['pick_ast'] == "UNDER":
                            print("CORRECT")
                            results[item] = proj[item]
                            results[item]['ast_res'] = "W"
                            correctPicks = correctPicks +1
                            correctAst = correctAst +1
                        else:
                            print("INCORRECT")
                            results[item] = proj[item]
                            results[item]['ast_res'] = "L"
                            incorrectPicks = incorrectPicks + 1
                            incorrectAst = incorrectAst +1

                    if pl['statistics']['reboundsTotal'] > -1 and 'fanduel_reb' in proj[item]:

                        if pl['statistics']['reboundsTotal'] > proj[item]['fanduel_reb'] and proj[item]['pick_reb'] == "OVER":
                            print("CORRECT")
                            results[item] = proj[item]
                            results[item]['ast_res'] = "W"
                            correctPicks = correctPicks +1
                            correctReb = correctReb + 1
                        elif pl['statistics']['reboundsTotal'] < proj[item]['fanduel_reb'] and proj[item]['pick_reb'] == "UNDER":
                            print("CORRECT")
                            results[item] = proj[item]
                            results[item]['reb_res'] = "W"
                            correctPicks = correctPicks +1
                            correctReb = correctReb + 1
                        else:
                            print("INCORRECT")
                            results[item] = proj[item]
                            results[item]['reb_res'] = "L"
                            incorrectPicks = incorrectPicks + 1
                            incorrectReb = incorrectReb + 1

                    if pl['statistics']['steals'] > -1 and 'fanduel_stl' in proj[item]:

                        if pl['statistics']['steals'] > proj[item]['fanduel_stl'] and proj[item]['pick_stl'] == "OVER":
                            print("CORRECT")
                            results[item] = proj[item]
                            results[item]['stl_res'] = "W"
                            correctPicks = correctPicks +1
                            correctStl = correctStl + 1
                        elif pl['statistics']['steals'] < proj[item]['fanduel_stl'] and proj[item]['pick_stl'] == "UNDER":
                            print("CORRECT")
                            results[item] = proj[item]
                            results[item]['stl_res'] = "W"
                            correctPicks = correctPicks +1
                            correctStl = correctStl + 1
                        else:
                            print("INCORRECT")
                            results[item] = proj[item]
                            results[item]['stl_res'] = "L"
                            incorrectPicks = incorrectPicks + 1
                            incorrectStl = incorrectStl + 1

                    if pl['statistics']['threePointersMade'] > -1 and 'fanduel_threes' in proj[item]:

                        if pl['statistics']['threePointersMade'] > proj[item]['fanduel_threes'] and proj[item]['pick_3s'] == "OVER":
                            print("CORRECT")
                            results[item] = proj[item]
                            results['threes_res'] = "W"
                            correctPicks = correctPicks +1
                            correctThree = correctThree + 1
                        elif pl['statistics']['threePointersMade'] < proj[item]['fanduel_threes'] and proj[item]['pick_3s'] == "UNDER":
                            print("CORRECT")
                            results[item] = proj[item]
                            results['threes_res'] = "W"
                            correctPicks = correctPicks +1
                            correctThree = correctThree + 1
                        else:
                            print("INCORRECT")
                            results[item] = proj[item]
                            results[item]['threes_res'] = "L"
                            incorrectPicks = incorrectPicks + 1
                            incorrectThree = incorrectThree + 1

                    if pl['statistics']['blocks'] > -1 and 'fanduel_blk' in proj[item]:

                        if pl['statistics']['blocks'] > proj[item]['fanduel_blk'] and proj[item]['pick_3s'] == "OVER":
                            print("CORRECT")
                            results[item] = proj[item]
                            results[item]['threes_res'] = "W"
                            #correctPicks = correctPicks +1
                        elif pl['statistics']['blocks'] < proj[item]['fanduel_blk'] and proj[item]['pick_blk'] == "UNDER":
                            print("CORRECT")
                            results[item] = proj[item]
                            results[item]['blk_res'] = "W"
                           # correctPicks = correctPicks +1
                        else:
                            print("INCORRECT")
                            results[item] = proj[item]
                            results[item]['blk_res'] = "L"
                           # incorrectPicks = incorrectPicks + 1


#print(results)
print("CORRECT: ", correctPicks)
print("INCORRECT: " ,incorrectPicks)
print("ACCURACY: ", correctPicks/(correctPicks+incorrectPicks))

results['overall'] = correctPicks/(correctPicks+incorrectPicks)
#results['overallCorrect'] = correctPicks
#results['overallIncorrect'] =incorrectPicks

print("CORRECT PTS: ", correctPts)
print("INCORRECT: " ,incorrectPts)
print("ACCURACY: ", correctPts/(correctPts+incorrectPts))

results['pts'] = correctPts/(correctPts+incorrectPts)
#results['correctPts'] = correctPts
#results['incorrectPts'] = incorrectPts

print("CORRECT AST: ", correctAst)
print("INCORRECT: " ,incorrectAst)
print("ACCURACY: ", correctAst/(correctAst+incorrectAst))

results['ast'] = correctAst/(correctAst+incorrectAst)
#results['correctAst'] = correctAst
#results['incorrectAst'] = incorrectAst

print("CORRECT REB: ", correctReb)
print("INCORRECT: " ,incorrectReb)
print("ACCURACY: ", correctReb/(correctReb+incorrectReb))

results['reb'] = correctReb/(correctReb+incorrectReb)
#results['correctReb'] = correctReb
#results['incorrectReb'] = incorrectReb

print("CORRECT 3: ", correctThree)
print("INCORRECT: " ,incorrectThree)
print("ACCURACY: ", correctThree/(correctThree+incorrectThree))

results['threes'] = correctThree/(correctThree+incorrectThree)
#results['correctThrees'] = correctThree
#results['incorrectThrees'] = incorrectThree

print("CORRECT Stl: ", correctStl)
print("INCORRECT: " ,incorrectStl)
print("ACCURACY: ", correctStl/(correctStl+incorrectStl))

results['stl'] = correctStl/(correctStl+incorrectStl)
#results['correctStl'] = correctStl
#results['incorrectStl'] = incorrectStl


#print("CORRECT Blk: ", correctBlk)
#print("INCORRECT: " ,incorrectBlk)
#print("ACCURACY: ", correctBlk/(correctBlk+incorrectBlk))

#results['blk'] = correctBlk/(correctBlk+incorrectBlk)


dt = yesterday()

dt = dt+".json"

with open('history/'+dt, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)


#print(df[0]['personId'])

#print(gameids)