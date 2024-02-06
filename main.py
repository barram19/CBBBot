from fuzzywuzzy import process
import numpy as np
from sklearn.linear_model import LinearRegression
import requests
import xgboost as xgb
import pandas as pd
from tabulate import tabulate

import pytz
from datetime import datetime, timezone

# Specify the name of the uploaded Excel file
excel_file_name = 'CBB_Teams.xlsx'

# Function to read the team name mapping from an Excel file
def read_team_mapping(file_name):
    team_mapping = pd.read_excel(file_name)
    return team_mapping

# Read team name mapping from the Excel file in Google Colab
team_mapping = read_team_mapping(excel_file_name)

# Function to find the standardized team name from the Excel file
def find_standardized_name(team_name, team_mapping):
    match = team_mapping.loc[team_mapping['TeamName'] == team_name, 'StandardizedTeamName'].values
    if len(match) > 0:
        return match[0]
    else:
        return None


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
        commence_time = game['commence_time']
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

    # Match team names in odds data with average scoring data using the Excel file
    df['matched_home_team'] = df['home_team'].apply(lambda x: find_standardized_name(x, team_mapping))
    df['matched_away_team'] = df['away_team'].apply(lambda x: find_standardized_name(x, team_mapping))

    # Merge average scoring data into the odds data based on matched team names
    df = pd.merge(df, df_points_for, left_on='matched_home_team', right_on='Team', how='left')
    df = pd.merge(df, df_points_for, left_on='matched_away_team', right_on='Team', how='left', suffixes=('_home', '_away'))

    # Match team names in odds data with average opponent points against data
    df['matched_home_team_against'] = df['home_team'].apply(lambda x: find_standardized_name(x, team_mapping))
    df['matched_away_team_against'] = df['away_team'].apply(lambda x: find_standardized_name(x, team_mapping))

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
    df = df.dropna(subset=['matched_home_team', 'matched_away_team', 'matched_home_team_against', 'matched_away_team_against'])

    # Display the combined DataFrame
    table = tabulate(df, headers='keys', tablefmt='pretty')
    print(table)

    # FIND THE VARIANCE BETWEEN LINE AND PREDICTED FINAL SCORE

    # Extract the relevant columns for prediction
    columns_for_home_team = ['2023_home', '2023_away']
    columns_for_away_team = ['2023_home_against', '2023_away_against']

    # Define the feature matrices for home and away teams
    X_home_team = df[columns_for_home_team].values.reshape(-1, len(columns_for_home_team))
    X_away_team = df[columns_for_away_team].values.reshape(-1, len(columns_for_away_team))

    # Combine feature matrices and target variable for dropping missing values
    columns_to_drop_na = columns_for_home_team + columns_for_away_team + ['point']
    df_combined = df[columns_to_drop_na]

   # Drop rows with missing values
    df_combined = df_combined.dropna()

    # Reset the index
    df_combined = df_combined.reset_index(drop=True)

    # Separate back into feature matrices and target variable
    X_home_team = df_combined[columns_for_home_team].values.reshape(-1, len(columns_for_home_team))
    X_away_team = df_combined[columns_for_away_team].values.reshape(-1, len(columns_for_away_team))
    y = df_combined['point'].values

    # Initialize separate linear regression models for home and away teams
    model_home_team = LinearRegression()
    model_away_team = LinearRegression()

    # Fit the models
    model_home_team.fit(X_home_team, y)
    model_away_team.fit(X_away_team, y)

    # Predict scores for each team
    predicted_home_scores = model_home_team.predict(X_home_team)
    predicted_away_scores = model_away_team.predict(X_away_team)

    # Create a new DataFrame for predictions
    prediction_df = pd.DataFrame({
        'Predicted_Score_Team_Home': predicted_home_scores,
        'Predicted_Score_Team_Away': predicted_away_scores
    })

    # Concatenate the prediction DataFrame with the original DataFrame
    df = pd.concat([df, prediction_df], axis=1)

    # Calculate an average predicted score for both teams
    df['Predicted_Average_Score'] = (df['Predicted_Score_Team_Home'] + df['Predicted_Score_Team_Away']) / 2

    # Display the predicted scores
    print(tabulate(df[['home_team', 'away_team', 'Predicted_Average_Score', 'point']], headers='keys', tablefmt='pretty'))
    

# Calculate variance between predicted and actual points
df['Variance'] = df['Predicted_Average_Score'] - df['point']

# Add a column indicating over or under the point metric
df['Over_Under'] = np.where(df['Variance'] > 0, 'Over', 'Under')

# Get current UTC time
current_utc_time = datetime.utcnow().replace(tzinfo=timezone.utc)

# Convert UTC time to CST
cst_timezone = pytz.timezone('US/Central')
current_cst_time = current_utc_time.astimezone(cst_timezone)

# Format the time as a string
formatted_time_utc = current_utc_time.strftime("%Y-%m-%dT%H:%M:%SZ")
formatted_time_cst = current_cst_time.strftime("%Y-%m-%d %H:%M:%S CST")

# Add current UTC date time to the dataframe
df['Current_UTC_Time'] = formatted_time_utc

# Convert 'commence_time' column to datetime
df['commence_time'] = pd.to_datetime(df['commence_time'], utc=True)

# Include results with more than 5-point variance either direction
filtered_df = df[(np.abs(df['Variance']) > 5) & (df['commence_time'] > current_utc_time)].copy()

# Convert 'commence_time' column to CST
filtered_df['Commence_Time_CST'] = filtered_df['commence_time'].dt.tz_convert(cst_timezone).dt.strftime("%Y-%m-%d %H:%M:%S CST")

# Display the filtered DataFrame with Predicted_Average_Score
print(tabulate(filtered_df[['home_team', 'away_team', 'Predicted_Average_Score', 'point', 'Variance', 'Over_Under', 'Commence_Time_CST']], headers='keys', tablefmt='pretty'))



import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Set up Google Sheets API credentials
credentials_info = {
    "type": "service_account",
    "project_id": "PROJECTIDHERE",
    "private_key_id": "PRIVATEKEYIDHERE",
    "private_key": "PRIVATEKEYHERE"
    "client_email": "SERVICEACCOUNTEMAILHERE",
    "client_id": "CLIENTIDHERE",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "CLIENTCERTHERE",
    "universe_domain": "googleapis.com"
}

# Set up Google Sheets API credentials
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(credentials_info, scope)
client = gspread.authorize(creds)

# Open the Google Sheet by title
spreadsheet_title = "INSERTSPREADSHEETTITLEHERE"
spreadsheet = client.open(spreadsheet_title)

# Access the specific worksheet within the spreadsheet
worksheet_title = "WORKSHEETTITLEHERE"
worksheet = spreadsheet.worksheet(worksheet_title)

# Convert filtered_df to a list of dictionaries
new_data = filtered_df[['home_team', 'away_team', 'Predicted_Average_Score', 'point', 'Variance', 'Over_Under', 'Commence_Time_CST']].to_dict(orient='records')

# Load existing data from the worksheet
existing_data = worksheet.get_all_records()

# Convert filtered_df to a list of dictionaries
new_data = filtered_df[['home_team', 'away_team', 'Predicted_Average_Score', 'point', 'Variance', 'Over_Under', 'Commence_Time_CST']].to_dict(orient='records')

# Check for duplicate entries based on home team, away team, and commence date
for entry in new_data:
    # Convert commence_time to string for comparison
    entry['Commence_Time_CST'] = str(entry['Commence_Time_CST'])
    
    # Check if the entry already exists
    is_duplicate = any(
        (entry['home_team'] == row['home_team'] and
         entry['away_team'] == row['away_team'] and
         entry['Commence_Time_CST'] == str(row['Commence_Time_CST']))
        for row in existing_data
    )
    
    # If not a duplicate, append the new entry to the worksheet
    if not is_duplicate:
        worksheet.append_row([entry['home_team'], entry['away_team'], entry['point'], entry['Predicted_Average_Score'],  entry['Variance'], entry['Over_Under'], entry['Commence_Time_CST']])
