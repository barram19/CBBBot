import numpy as np
from sklearn.linear_model import LinearRegression
import requests
import xgboost as xgb
import pandas as pd
from tabulate import tabulate
from fuzzywuzzy import process


# Set API endpoint URL and API key for OddsData
url = 'https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds/?apiKey=00ab27442da4a2b1f8460c5c70d0b3d8&regions=us&markets=h2h,spreads,totals&oddsFormat=american&bookmakers=fanduel'
api_key = '00ab27442da4a2b1f8460c5c70d0b3d8'

# Set headers for the request
headers = {
    'Content-Type': 'application/json',
    'x-api-key': api_key,
}

# Make the request and get the response data
response = requests.get(url, headers=headers)

if response.status_code == 200:
    odds_data = response.json()

    # Extract the necessary information from the odds data
    game_data = []
    for game in odds_data:
        home_team = game['home_team']
        away_team = game['away_team']
        commence_time = game['commence_time'][:10]  # Extract the date from the commence_time field
        bookmakers = game['bookmakers']
        for bookmaker in bookmakers:
            markets = bookmaker['markets']
            for market in markets:
                if market['key'] == 'h2h':
                    outcomes = market['outcomes']
                    home_odds = outcomes[0]['price']
                    away_odds = outcomes[1]['price']
                elif market['key'] == 'totals':
                    outcomes = market['outcomes']
                    point = outcomes[0]['point']
                    over_odds = outcomes[0]['price']
                    under_odds = outcomes[1]['price']
                    if point is not None:
                        game_data.append([home_team, away_team, commence_time, home_odds, away_odds, point, over_odds,
                                          under_odds])

    # Get average points for
    url_points_for = 'https://www.teamrankings.com/ncaa-basketball/stat/points-per-game'
    df_points_for = pd.read_html(url_points_for)[0]

    # Get average points against
    url_points_against = 'https://www.teamrankings.com/ncaa-basketball/stat/opponent-points-per-game'
    df_points_against = pd.read_html(url_points_against)[0]

    # Convert game_data into a DataFrame
    columns = ['home_team', 'away_team', 'commence_time', 'home_odds', 'away_odds', 'point', 'over_odds', 'under_odds']
    df = pd.DataFrame(game_data, columns=columns)

    # Function to find the best match for a team name using fuzzy matching
    def find_best_match(team_name, options):
        match_info = process.extractOne(team_name, options)
        if match_info[1] >= 90:  # Adjust the threshold as needed
            return match_info[0]
        else:
            return None

    # Match team names in odds data with average scoring data
    df['matched_home_team'] = df['home_team'].apply(lambda x: find_best_match(x, df_points_for['Team']))
    df['matched_away_team'] = df['away_team'].apply(lambda x: find_best_match(x, df_points_for['Team']))

    # Merge average scoring data into the odds data based on matched team names
    df = pd.merge(df, df_points_for, left_on='matched_home_team', right_on='Team', how='left')
    df = pd.merge(df, df_points_for, left_on='matched_away_team', right_on='Team', how='left', suffixes=('_home', '_away'))

    # Match team names in odds data with average opponent points against data
    df['matched_home_team_against'] = df['home_team'].apply(lambda x: find_best_match(x, df_points_against['Team']))
    df['matched_away_team_against'] = df['away_team'].apply(lambda x: find_best_match(x, df_points_against['Team']))

    # Merge average opponent points against data into the odds data based on matched team names
    df = pd.merge(df, df_points_against, left_on='matched_home_team_against', right_on='Team', how='left',
                  suffixes=('_home_against', '_away_against'))
    df = pd.merge(df, df_points_against, left_on='matched_away_team_against', right_on='Team', how='left',
                  suffixes=('_home_against', '_away_against'))

   # Drop unnecessary columns
columns_to_drop = ['Rank_home', '2022_home', 'Rank_away', '2022_away', 'Rank_home_against', '2022_home_against',
                   'Rank_away_against', '2022_away_against']

df.drop(columns_to_drop, axis=1, inplace=True)


# Filter out rows where team match was not found
df = df.dropna(subset=['matched_home_team', 'matched_away_team','matched_home_team_against', 'matched_away_team_against'])

# Display the combined DataFrame
table = tabulate(df, headers='keys', tablefmt='pretty')
print(table)

# FIND THE VARIANCE BETWEEN LINE AND PREDICTED FINAL SCORE



# ... (your existing code)


# Extract the relevant columns for prediction
columns_for_home_team = ['2023_home', '2023_away']
columns_for_away_team = ['2023_home_against', '2023_away_against']

# Define the feature matrices for home and away teams
X_home_team = df[columns_for_home_team].values.reshape(-1, len(columns_for_home_team))
X_away_team = df[columns_for_away_team].values.reshape(-1, len(columns_for_away_team))

# Initialize separate linear regression models for home and away teams
model_home_team = LinearRegression()
model_away_team = LinearRegression()

# Fit the models
model_home_team.fit(X_home_team, df['point'])
model_away_team.fit(X_away_team, df['point'])

# Predict scores for each team
df['Predicted_Score_Team_Home'] = model_home_team.predict(X_home_team)
df['Predicted_Score_Team_Away'] = model_away_team.predict(X_away_team)

# Calculate an average predicted score for both teams
df['Predicted_Average_Score'] = (df['Predicted_Score_Team_Home'] + df['Predicted_Score_Team_Away']) / 2

# Display the predicted scores
print(tabulate(df[['home_team', 'away_team', 'Predicted_Average_Score', 'point']], headers='keys', tablefmt='pretty'))

# Calculate variance between predicted and actual points
df['Variance'] = df['Predicted_Average_Score'] - df['point']

# Add a column indicating over or under the point metric
df['Over_Under'] = np.where(df['Variance'] > 0, 'Over', 'Under')

# Filter results with more than 5-point variance
filtered_df = df[np.abs(df['Variance']) > 5]

# Display filtered variance information with over/under column
print(tabulate(filtered_df[['home_team', 'away_team', 'Variance', 'Over_Under']], headers='keys', tablefmt='pretty'))
