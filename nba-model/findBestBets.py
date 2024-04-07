import json

with open('data.json') as json_file:
    data = json.load(json_file)
 
ptsweights = {}
astweights = {}
rebweights = {}
for d in data:
    if data[d] != None:
        if "fanduel_pts" in data[d] :
            # print(data[d])
            if data[d]['fanduel_pts'] != 0:
                ptsweights[d] = data[d]["PTS"] - data[d]["fanduel_pts"]
            if data[d]['fanduel_ast'] != 0:
                astweights[d] = data[d]["AST"] - data[d]["fanduel_ast"]
            if data[d]['fanduel_reb'] != 0:
                rebweights[d] = data[d]["REB"] - data[d]["fanduel_reb"]


print(len(ptsweights))

count = 0
maxnames = []
minnames = []

bets = {}
while count < 5:  
    mi = 100
    ma = -100
    maxname = ""
    minname = ""
    for p in ptsweights:
        if ptsweights[p] > ma:
            ma = ptsweights[p]
            if p not in maxnames:
                maxname = p
        
        elif ptsweights[p] < mi:
            mi = ptsweights[p]
            if p not in minnames:
                minname = p
            #minname = p
        
    #print(minname, mi, maxname, ma)

    maxnames.append(maxname)
    minnames.append(minname)
    count = count + 1


print("MAX PTS: " , maxnames)
print("MIN PTS: ", minnames)

bets['maxpts'] = maxnames
bets['minpts'] = minnames

count = 0
maxnames = []
minnames = []
while count < 5:  
    mi = 100
    ma = -100
    maxname = ""
    minname = ""
    for p in astweights:
        if astweights[p] > ma:
            ma = astweights[p]
            if p not in maxnames:

                maxname = p
        
        elif astweights[p] < mi:
            mi = astweights[p]
            if p not in minnames:
                minname = p
            #minname = p
        
    #print(minname, mi, maxname, ma)

    maxnames.append(maxname)
    minnames.append(minname)
    count = count + 1

print("MAX AST: " , maxnames)
print("MIN AST: ", minnames)

bets['maxast'] = maxnames
bets['minast'] = minnames


count = 0
maxnames = []
minnames = []
while count < 5:  
    mi = 100
    ma = -100
    maxname = ""
    minname = ""
    for p in rebweights:
        if rebweights[p] > ma:
            ma = rebweights[p]
            if p not in maxnames:
                maxname = p
        
        elif rebweights[p] < mi:
            mi = rebweights[p]
            if p not in minnames:
                minname = p
            #minname = p
        
   # print(minname, mi, maxname, ma)

    maxnames.append(maxname)
    minnames.append(minname)
    count = count + 1
print("MAX REB: " , maxnames)
print("MIN REB: ", minnames)

bets['maxreb'] = maxnames
bets['minreb'] = minnames

print(bets)

with open('bestBets.json', 'w', encoding='utf-8') as f:
        json.dump(bets, f, ensure_ascii=False, indent=4)


def test(d):
    m = max(d, key=d.get)
    mi =  min(d, key=d.get)
    print(list(d.keys())[list(d.values()).index(m)])
    return m, mi

#test(ptsweights)