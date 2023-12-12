import json
 


with open('projections.json') as json_file:
    data = json.load(json_file)
 
    # Print the type of data variable
    print("Type:", type(data))

with open('lines.json') as json_file:
    lines = json.load(json_file)
 
    # Print the type of data variable
    print("Type:", type(data))

count = 0

dic = {}

li = []


for line in lines:
    for d in data[0]:
        #print(data[0][d])
        #print("DATA [0]: ", data[0])
        dic[d] = data[0][d]
        #print(type(data[0]))
        if d == line["name"]:
            # print("LINE: ", d)
            #
            #dic[d].update({"fanduel_pts": line['fanduel_pts']})
            print(type(dic[d]))
            if type(dic[d]) != dict:
                print(dic[d])
            else:
                
                if line['fanduel_pts'] != "" or line['fanduel_pts'] != None:
                    try:
                        if line['fanduel_pts'] == None:
                            dic[d]['fanduel_pts'] = 0
                        else:
                            dic[d]['fanduel_pts'] = float(line['fanduel_pts'])
                            print( float(line['fanduel_pts']))
                    except ValueError:      
                        dic[d]['fanduel_pts'] = 0
                else:
                    dic[d]['fanduel_pts'] = 0

                if line['fanduel_reb'] != "" or line['fanduel_reb'] != None:
                    try:
                        if line['fanduel_reb'] == None:
                            dic[d]['fanduel_reb'] = 0
                        else:
                            dic[d]['fanduel_reb'] = float(line['fanduel_reb'])
                    except ValueError:      
                        dic[d]['fanduel_reb'] = 0
                else:
                    dic[d]['fanduel_reb'] = 0

                #dic[d]['fanduel_reb'] = float(line['fanduel_reb'])
                
                if line['fanduel_ast'] != "" or line['fanduel_ast'] != None:
                    try:
                        if line['fanduel_ast'] == None:
                            dic[d]['fanduel_ast'] = 0
                        else:
                            dic[d]['fanduel_ast'] = float(line['fanduel_ast'])
                    except ValueError:
                        dic[d]['fanduel_ast'] = 0
                else:
                    dic[d]['fanduel_ast'] = 0
                
                #dic[d]['fanduel_ast'] = float(line['fanduel_ast'])

                if line['fanduel_threes'] != "" or line['fanduel_threes'] != None:
                    try:
                        if line['fanduel_threes'] ==None:
                            dic[d]['fanduel_threes'] = 0
                        else:
                            dic[d]['fanduel_threes'] = float(line['fanduel_threes'])
                    except ValueError:
                        dic[d]['fanduel_threes'] = 0
                else:
                    dic[d]['fanduel_threes'] = 0


                if line['fanduel_blk'] != "" or line['fanduel_blk'] != None:
                    try:
                        if line['fanduel_blk'] == None:
                            dic[d]['fanduel_blk'] = 0
                        else:
                            dic[d]['fanduel_blk'] = float(line['fanduel_blk'])
                    except ValueError:
                        dic[d]['fanduel_blk'] = 0

                else:
                    dic[d]['fanduel_blk'] = 0

                #dic[d]['fanduel_threes'] = float(line['fanduel_threes'])
                #dic[d]['fanduel_blk'] = float(line['fanduel_blk'])

                if line['fanduel_stl'] != "" or line['fanduel_stl']!= None:
                    try:
                        if line['fanduel_stl'] == None:
                            dic[d]['fanduel_stl'] = 0
                        else:
                            dic[d]['fanduel_stl'] = float(line['fanduel_stl'])
                    except ValueError:
                        dic[d]['fanduel_stl'] = 0

                else:
                    dic[d]['fanduel_stl'] = 0

                #dic[d]['fanduel_stl'] = float(line['fanduel_stl'])

                if dic[d]["PTS"] > dic[d]['fanduel_pts']:
                    dic[d]['pick_pts'] = "OVER"
                else:
                    dic[d]['pick_pts'] = "UNDER"

                if dic[d]["REB"] > dic[d]['fanduel_reb']:
                    dic[d]['pick_reb'] = "OVER"
                else:
                    dic[d]['pick_reb'] = "UNDER"

                if dic[d]["AST"] > dic[d]['fanduel_ast']:
                    dic[d]['pick_ast'] = "OVER"
                else:
                    dic[d]['pick_ast'] = "UNDER"

                if dic[d]["FG3M"] > dic[d]['fanduel_threes']:
                    dic[d]['pick_3s'] = "OVER"
                else:
                    dic[d]['pick_3s'] = "UNDER"

                if dic[d]["BLK"] > dic[d]['fanduel_blk']:
                    dic[d]['pick_blk'] = "OVER"
                else:
                    dic[d]['pick_blk'] = "UNDER"

                if dic[d]["STL"] > dic[d]['fanduel_stl']:
                    dic[d]['pick_stl'] = "OVER"
                else:
                    dic[d]['pick_stl'] = "UNDER"

print(dic)
with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(dic, f, ensure_ascii=False, indent=4)

##print(data)
#print(lines)

