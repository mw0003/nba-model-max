from turtle import pos, position
from nba_api.stats.endpoints import teamvsplayer, matchupsrollup,teamestimatedmetrics, playervsplayer, defensehub

import json
from statistics import median


################################
################################
###### FUNCTION TEST FILE ######
################################
################################

with open('defensevposition.json') as json_file:
        defensestats = json.load(json_file)

teamName = "OKC"
posit = "PG"
ppgavg = []
astavg = []
rebavg = []


for key in defensestats:

    team = key['TEAM'].split(" ")[0]
    position = key['POSITION'].split(" ")[0]

    if posit == position:
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

    if posit == position and team == teamName:
        
        ppg = float(key['PTS'].split(" ")[0])
        ast = float(key['AST'].split(" ")[0])
        reb = float(key['REB'].split(" ")[0])

        ptsRatio = ppg/leagueAvgPts
        astRatio = ast/leagueAvgAst
        rebRatio = reb/leagueAvgReb

        print(ptsRatio, astRatio, rebRatio)
        print("FOUND")

    