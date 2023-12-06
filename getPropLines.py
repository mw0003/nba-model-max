import site
from bs4 import BeautifulSoup
import requests
import pandas as pd
from nba_api.stats.static import players
import json

def substring_after(s, delim):
    return s.partition(delim)[2]

url = "https://www.rotowire.com/betting/nba/player-props.php?book=fanduel"
soup = BeautifulSoup(requests.get(url).text, "html.parser")

print(soup)
#data = json.loads(soup.find('script', type='application/ld+json').text)
#print(data)
#starters class
#table = soup.find_all(class_= 'prop-table')

#script_text = soup.find('script')
#site_json=json.loads(soup.text)

result = [scr.get_text() for scr in soup.find_all('script') ]
#res = soup.find_all('script')

##Holds lines 

dataIndex = result[4].index("data: [{")
dataIndex = dataIndex+7

string = result[4][dataIndex:]


import re
names = [m.start() for m in re.finditer('name', string = result[4][dataIndex:])]
lines = [m.start() for m in re.finditer('fanduel_pts', string = result[4][dataIndex:])]

linesArr = []
print(lines)
for line in lines:
    #print(len(string))
    #print(type(line))
    if (type(line) == int):
        if len(string) > line+14:
            endIndex = string.find(',', line)
           # lineUpd = string.find('"', line+14)
            lineUpd = line+14
           # print(endIndex)
           # print(string[lineUpd:endIndex])
            linesArr.append(string[lineUpd:endIndex])

print(linesArr)
   
#for name in names: 
    #fullName = 

#for line in lines:


#print(lines)

#endIndex = string.index("}],")



#print(result[4][endIndex:endIndex+10])

#print(result[4][dataIndex:endIndex])
#site_json=json.loads(result[4][dataIndex:endIndex])
#print(site_json)

#print(result[4][dataIndex:endIndex+2])

#print(string)
#data = substring_after(result[4][dataIndex:], '}],')
#print(data)





#printing for entrezgene, do the same for name and symbol
#print([d.get('draftkings_reb') for d in site_json['hits'] if d.get('draftkings_reb')])

#stats = soup.find_all(class_= 'draftkings_reb')
#print(script_text)
#print(soup.find_all(string= 'data: [{'))
#names = soup.find_all('a', class_='odds-table-entity-link')
#odds = soup.find_all('span', class_='odds-table-odds-link__val')

#print(names)
#print(odds)
#full = [x.find('span').text for x in names]
#odds2 = [x.find('a').text for x in odds]

#print(full)
#print(odds2)

"""
names = [x.find('a')['title'] for x in lineups]
teams = sum([[x.text] * 5 for x in soup.find_all(class_='lineup__abbr')], [])

print(names)
for name in names:
    
    p = players.find_players_by_full_name(name)
    print(p)

df = pd.DataFrame(zip(names, teams, positions))
jsonLineups = df.to_json(orient='split')


#Get injured Players 
playersOut = soup.find_all(class_='is-pct-play-0')
names2 = [x.find('a')['title'] for x in playersOut]

df = pd.DataFrame(zip(names2))

json_object = df.to_json(orient='split')
print(json_object)


#Write out/in to file

with open('lineups.json', 'w', encoding='utf-8') as f:
    json.dump(jsonLineups, f, ensure_ascii=False, indent=4) 

"""