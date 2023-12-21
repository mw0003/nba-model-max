

import site
from turtle import pos, position
from bs4 import BeautifulSoup
import requests
import pandas as pd
from nba_api.stats.static import players
import json

url = "https://hashtagbasketball.com/nba-defense-vs-position"
soup = BeautifulSoup(requests.get(url).text, "html.parser")

#data = json.loads(soup.find('script', type='application/ld+json').text)
#print(data)
#starters class
#table = soup.find_all(class_= 'prop-table')

#script_text = soup.find('script')
#site_json=json.loads(soup.text)
lineups = soup.find_all(class_='table--statistics')

positions = [x.find('tr').text for x in lineups]
r =[x.find('td') for x in positions]

print(r)