import pandas as pd
import json

# Load data from JSON file
with open('data.json', 'r') as file:
    data = json.load(file)

# Create DataFrame
df = pd.DataFrame(data).T.reset_index()
df.columns = ['Player'] + list(df.columns[1:])

# Keep only the desired columns
desired_columns = ['Player', 'PTS', 'REB', 'AST', 'fanduel_pts', 'fanduel_reb', 'fanduel_ast', 
                   'pick_pts', 'pick_reb', 'pick_ast']
df = df[desired_columns]

# Convert relevant columns to numeric type
numeric_columns = ['PTS', 'REB', 'AST', 'fanduel_pts', 'fanduel_reb', 'fanduel_ast']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Round all decimals to the tenths place
df = df.round(decimals=1)

# Add columns for absolute differences
df['abs_diff_pts'] = abs(df['PTS'] - df['fanduel_pts'])
df['abs_diff_reb'] = abs(df['REB'] - df['fanduel_reb'])
df['abs_diff_ast'] = abs(df['AST'] - df['fanduel_ast'])

# Calculate percentage differences as rounded absolute values
df['perc_diff_pts'] = round((abs(df['PTS'] - df['fanduel_pts']) / df['fanduel_pts']) * 100, 1)
df['perc_diff_reb'] = round((abs(df['REB'] - df['fanduel_reb']) / df['fanduel_reb']) * 100, 1)
df['perc_diff_ast'] = round((abs(df['AST'] - df['fanduel_ast']) / df['fanduel_ast']) * 100, 1)

# Drop rows with NaN values
df = df.dropna()

# Filter out rows where Fanduel column has a value of 0
df_pts = df[df['fanduel_pts'] != 0][['Player', 'PTS', 'fanduel_pts', 'abs_diff_pts', 'perc_diff_pts', 'pick_pts']]
df_reb = df[df['fanduel_reb'] != 0][['Player', 'REB', 'fanduel_reb', 'abs_diff_reb', 'perc_diff_reb', 'pick_reb']]
df_ast = df[df['fanduel_ast'] != 0][['Player', 'AST', 'fanduel_ast', 'abs_diff_ast', 'perc_diff_ast', 'pick_ast']]

# Sort by absolute percentage difference and select top 8
df_pts = df_pts.sort_values(by='perc_diff_pts', ascending=False).head(8)
df_reb = df_reb.sort_values(by='perc_diff_reb', ascending=False).head(8)
df_ast = df_ast.sort_values(by='perc_diff_ast', ascending=False).head(8)

# Collect unique players from final DataFrames
final_players = set(df_pts['Player']).union(set(df_reb['Player'])).union(set(df_ast['Player']))

# Determine max and min players based on pick columns
max_pts_players = df_pts[df_pts['pick_pts'] == 'OVER']['Player'].tolist()
min_pts_players = df_pts[df_pts['pick_pts'] == 'UNDER']['Player'].tolist()
max_reb_players = df_reb[df_reb['pick_reb'] == 'OVER']['Player'].tolist()
min_reb_players = df_reb[df_reb['pick_reb'] == 'UNDER']['Player'].tolist()
max_ast_players = df_ast[df_ast['pick_ast'] == 'OVER']['Player'].tolist()
min_ast_players = df_ast[df_ast['pick_ast'] == 'UNDER']['Player'].tolist()

# Load existing bestBets.json file and clear existing data
existing_data = {
    "maxpts": [],
    "minpts": [],
    "maxast": [],
    "minast": [],
    "maxreb": [],
    "minreb": []
}

# Update max/min players in the existing_data dictionary
existing_data['maxpts'] = max_pts_players
existing_data['minpts'] = min_pts_players
existing_data['maxast'] = max_ast_players
existing_data['minast'] = min_ast_players
existing_data['maxreb'] = max_reb_players
existing_data['minreb'] = min_reb_players

# Print existing_data before exporting to JSON
print("Data to be dumped into bestBets.json:")
print(existing_data)

# Export updated data to bestBets.json
with open('bestBets.json', 'w') as best_bets_file:
    json.dump(existing_data, best_bets_file, indent=4)

# Display DataFrames
print("\nDataFrame for PTS:")
print(df_pts)
print("\nDataFrame for REB:")
print(df_reb)
print("\nDataFrame for AST:")
print(df_ast)
