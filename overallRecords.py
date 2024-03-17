from datetime import datetime
import json
import os, glob


path = "./history"
overallCorrect = 0
count = 0
correctPts = 0
incorrectPts = 0
correctAst = 0
incorrectAst = 0
correctReb = 0
incorrectReb =0
correct3 = 0
incorrect3 = 0
correctStl = 0
incorrectStl =0
accuracyOverTime = {}
pointsOverTime = {}
rebOverTime = {}
astOverTime = {}
threeOverTime = {}
stlOverTime = {}
bbPts = 0
bbIPts = 0
bbReb = 0
bbIReb = 0
bbAst = 0
bbIAst = 0

allObj = {"players":{

}}
minObj ={}
maxObj ={}
files = []

for filename in glob.glob(os.path.join(path, '*.json')):
    files.append(filename)
    with open(filename) as json_file:
        data = json.load(json_file)
        date = filename[10:-5]
        accuracyOverTime[date] = data["overall"]
        pointsOverTime[date] = data["pts"]
        rebOverTime[date] = data["reb"]
        astOverTime[date] = data["ast"]
        threeOverTime[date] = data["threes"]
        stlOverTime[date] = data['stl']
        if "bestBetsPtsCorrect" in data:
            bbPts += data['bestBetsPtsCorrect']
            bbIPts += data['bestBetsPtsInCorrect']
            bbIAst += data['bestBetsAstIncorrect']
            bbAst += data['bestBetsAst']
            bbReb += data['bestBetsReb']
            bbIReb += data['bestBetsRebIncorrect']

        if "overallCorrect" not in data:
            count = 0
            for r in data:
                if type(data[r]) == float or type(data[r]) == int:
                    print("float found")
                else:
                    if type(data[r]) != float or type(data[r]) != int:

                        if "fanduel_pts" in data[r] and data[r]["PTS"] > -1:
                            name = r
                            ptsDiff = data[r]['fanduel_pts'] - data[r]['PTS']
                            ptsres = ''
                            astres = ''
                            rebres = ''
                            if "pts_res" in data[r]:
                                ptsres = data[r]["pts_res"]
                            if "ast_res" in data[r]:
                                astres = data[r]["ast_res"]
                            if "reb_res" in data[r]:
                                rebres = data[r]["reb_res"]
                            if "fanduel_ast" in data[r]:
                                astDiff = data[r]['fanduel_ast'] - data[r]['AST']
                            if "fanduel_reb" in data[r]:
                                rebDiff = data[r]['fanduel_reb'] - data[r]['REB']
                            if rebDiff and astDiff:
                                print("ptsDiff" , type(ptsDiff))
                                obj = {
                                    name:{
                                    "ptsDiff": ptsDiff,
                                    "astDiff": astDiff,
                                    "rebDiff": rebDiff,
                                    'reb_res': rebres,
                                    'ast_res': astres,
                                    'pts_res': ptsres,
                                    "name": name
                                    }
                                    
                                }
                                allObj['players'].update(obj)
                        if "pts_res" in data[r]:
                            if data[r]["pts_res"] == "W":
                                correctPts +=1 
                            else:
                                incorrectPts +=1
                        if "ast_res" in data[r]:
                            if data[r]["ast_res"] == "W":
                                correctAst +=1 
                            else:
                                incorrectAst +=1
                        if "reb_res" in data[r]:
                            if data[r]["reb_res"] == "W":
                                correctReb +=1 
                            else:
                                incorrectReb +=1
                        if "threes_res" in data[r]:
                            if data[r]["threes_res"] == "W":
                                correct3 +=1 
                            else:
                                incorrect3 +=1
                        if "stl_res" in data[r]:
                            if data[r]["stl_res"] == "W":
                                correctStl +=1 
                            else:
                                incorrectStl +=1
       #         if type(d) != "string":
        #            print (d)
        #        elif 'pts_res' in data[d]:  
        #            print(data[d]['pts_res'])

        # Print the type of data variable
        count = count +1
        print("Type:", type(data))
    
    #print(allObj)
    for i in range(0,3):
        
        el = max(allObj['players'].values(), key=lambda ev:ev['rebDiff'])
        name = el['name']
        el = allObj['players'].pop(name)
        obj = {
            name:{

            }
        }
        obj[name].update(el)
        maxObj.update(obj)
        print(el)
    for i in range(0,3):
        
        el = min(allObj['players'].values(), key=lambda ev:ev['rebDiff'])
        name = el['name']
        el = allObj['players'].pop(name)
        obj = {
            name:{

            }
        }
        obj[name].update(el)
        minObj.update(obj)
        print(el)
   
print("max" , maxObj)
print("min" , minObj)

pts_result_max = 0
pts_result_total = 0
for key in maxObj:
    result = maxObj[key]['reb_res']
    if result == 'W':
        pts_result_max = pts_result_max +1
    pts_result_total = pts_result_total + 1
for key in minObj:
    result = minObj[key]['reb_res']
    if result == 'W':
        pts_result_max = pts_result_max +1
    pts_result_total = pts_result_total + 1

print('pts correct: ', pts_result_max/pts_result_total)
print("total ", pts_result_total)

#print()

#sys.sleep()

total = correctPts + incorrectPts + correctAst + incorrectAst + correctReb +incorrectReb+incorrect3 + correct3 +correctStl +incorrectStl
correct = correctPts +correctAst+correctReb+correct3+correctStl

print(correct/total)
print(accuracyOverTime)

accuracyOverTime = sorted(accuracyOverTime.items(), key = lambda x:datetime.strptime(x[0], '%Y-%m-%d'), reverse=True)
pointsOverTime = sorted(pointsOverTime.items(), key = lambda x:datetime.strptime(x[0], '%Y-%m-%d'), reverse=True)
astOverTime = sorted(astOverTime.items(), key = lambda x:datetime.strptime(x[0], '%Y-%m-%d'), reverse=True)
rebOverTime = sorted(rebOverTime.items(), key = lambda x:datetime.strptime(x[0], '%Y-%m-%d'), reverse=True)
threeOverTime = sorted(threeOverTime.items(), key = lambda x:datetime.strptime(x[0], '%Y-%m-%d'), reverse=True)
stlOverTime = sorted(stlOverTime.items(), key = lambda x:datetime.strptime(x[0], '%Y-%m-%d'), reverse=True)

overall = {}
overall["totalAcc"] = correct/total
overall["totalPicks"] = total
overall['ptsAcc'] = correctPts/(incorrectPts+correctPts)
overall['accOvertime'] = accuracyOverTime
overall['astAcc'] = correctAst/(incorrectAst+correctAst)
overall['rebAcc'] = correctReb/(incorrectReb+correctReb)
overall['threeAcc'] = correct3/(incorrect3+correct3)
overall['stlAcc'] = correctStl/(incorrectStl+correctStl)
overall['ptsOvertime'] = pointsOverTime
overall['rebOverTime'] = rebOverTime
overall['astOverTime'] = astOverTime
overall['threeOverTime'] =threeOverTime
overall['stlOverTime'] =stlOverTime
overall['bbPtsCorrect'] =bbPts
overall['bbPtsInCorrect'] =bbIPts
overall['bbAstCorrect'] =bbAst
overall['bbAstInCorrect'] =bbIAst
overall['bbRebCorrect'] =bbReb
overall['bbRebInCorrect'] =bbIReb

print(files)
with open('overall.json', 'w', encoding='utf-8') as f:
        json.dump(overall, f, ensure_ascii=False, indent=4)

#overallCorrect = overallCorrect/count

#print(overallCorrect)





