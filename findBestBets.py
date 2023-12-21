import json

with open('data.json') as json_file:
    data = json.load(json_file)
 
ptsweights = {}
astweights = {}
rebweights = {}
for d in data:
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
while count < 5:  
    mi = 100
    ma = -100
    maxname = ""
    minname = ""
    for p in ptsweights:
        if ptsweights[p] > ma:
            ma = ptsweights[p]
            if p in maxnames:
                print("none")
            else:
                print( p)
                maxname = p
        
        elif ptsweights[p] < mi:
            mi = ptsweights[p]
            if p in minnames:
                
                print("none")
            else:
                print( p)
                minname = p
            #minname = p
        
    print(minname, mi, maxname, ma)

    maxnames.append(maxname)
    minnames.append(minname)
    count = count + 1


print(maxnames)
print(minnames)
def test(d):
    m = max(d, key=d.get)
    mi =  min(d, key=d.get)
    print(list(d.keys())[list(d.values()).index(m)])
    return m, mi

#test(ptsweights)