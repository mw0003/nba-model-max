import requests

r = requests.get('https://www.bovada.lv/services/sports/event/v2/events/A/description/basketball/nba/los-angeles-lakers-sacramento-kings-201811102200?lang=en')

player_props = [
    t["markets"]
    for t in r.json()[0]["events"][0]["displayGroups"]
    if t["description"] == "Player Props"
]
specific_player = [
    t
    for t in player_props[0]
    if t["description"] == "Total Points - LeBron James (LAL)"
]
print(specific_player)