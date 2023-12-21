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
for filename in glob.glob(os.path.join(path, '*.json')):
    with open(filename) as json_file:
        data = json.load(json_file)
        date = filename[10:-5]
        accuracyOverTime[date] = data["overall"]
        pointsOverTime[date] = data["pts"]
        rebOverTime[date] = data["reb"]
        astOverTime[date] = data["ast"]
        threeOverTime[date] = data["threes"]
        stlOverTime[date] = data['stl']
        if "overallCorrect" not in data:
            for r in data:
                if type(data[r]) != float:
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

total = correctPts + incorrectPts + correctAst + incorrectAst + correctReb +incorrectReb+incorrect3 + correct3 +correctStl +incorrectStl
correct = correctPts +correctAst+correctReb+correct3+correctStl

print(correct/total)
print(accuracyOverTime)

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

with open('overall.json', 'w', encoding='utf-8') as f:
        json.dump(overall, f, ensure_ascii=False, indent=4)

#overallCorrect = overallCorrect/count

#print(overallCorrect)





