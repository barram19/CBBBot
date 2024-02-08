from fuzzywuzzy import process
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
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

    # Define the gradient boosting models for home and away teams
    model_home_team = GradientBoostingRegressor()
    model_away_team = GradientBoostingRegressor()

    # Fit the gradient boosting models
    model_home_team.fit(X_home_team, y)
    model_away_team.fit(X_away_team, y)

    # Predict scores for each team using gradient boosting models
    predicted_home_scores = model_home_team.predict(X_home_team)
    predicted_away_scores = model_away_team.predict(X_away_team)

    # Ensure lengths of predicted scores are consistent
    min_len = min(len(predicted_home_scores), len(predicted_away_scores))
    predicted_home_scores = predicted_home_scores[:min_len]
    predicted_away_scores = predicted_away_scores[:min_len]

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
filtered_df = df[(np.abs(df['Variance']) > 10) & (df['commence_time'] > current_utc_time)].copy()

# Convert 'commence_time' column to CST
filtered_df['Commence_Time_CST'] = filtered_df['commence_time'].dt.tz_convert(cst_timezone).dt.strftime("%Y-%m-%d %H:%M:%S CST")

# Display the filtered DataFrame with Predicted_Average_Score
print(tabulate(filtered_df[['home_team', 'away_team', 'Predicted_Average_Score', 'point', 'Variance', 'Over_Under', 'Commence_Time_CST']], headers='keys', tablefmt='pretty'))



import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import requests


# Set up Google Sheets API credentials
credentials_info = {
    "type": "service_account",
    "project_id": "cbbbot-413503",
    "private_key_id": "215d90539c7064779207f36f2c7efc3b8fe5205a",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC1LY0tVwJVpwUL\nG2gknmv75prag86+VUplFTONLtVRvDLfLxIJNv848tS4vkg3/8iui744OlqjGsq7\ncppqP804qrIv+oWeEAj3K8UM12yiqRSnW81OHYs8LalcLdYF8FxEkwNHdrHx+wr3\nZeHiDTn3K40hNlszNhy/azaqmKt19iDjts85FQKxW0CvdMs9CRy89BgJSTR6O0j9\ni5e9xMiwbMBvhOezsqIdLKQECh5l0X/lZFo0URhXHT9K11myYtWWrHvfBVip8h5V\nm0mCYVLVyBv0FIDX2I+nx9TVdpEUAKLfHF6nBSEaQ1cjl4Xo69jsAxodz/muccts\n5Ybtsol3AgMBAAECggEAKB8TQXQgNzmKW2BRWrKedSUnK7a+pNWcaPAd/2jcooIL\nvfLdip7cPA9CXjr9ITGKjmSx1h/ODIqVVJdXbKn+V0sttIRE7LDeW2Yc9/AIxait\nwzYILAFM4SG3fItF9wC4XhM0SbIWS+DtF8Y/FGEbcgn58d3oqlmUWity6qpuZeug\nxkqGB+BZoIFUtdETcOaFwiRh4leQRfvK6kAZ/MOljRqFEzo++bRXQyrvYQX8sSNn\nGj68rzYC5YtwxSAWBOHu58tP5tIEhdK5oNI69SbDWY7iJpmBc3J4sE1fDK18pm79\naUfRHtSIoc8p8J7fm0u4XwZvrZfgYhp83KQHmXuqAQKBgQDozlFEL5mQC2e+17p/\nbbSNeMhpHagHkVItV6uA0qQwyv1dRvu+4XTIu7KC24/q6z2B0QmUsWE4rU0qXP4H\nwxfVTV6s9t1EZqTJ0usJsgxS+nJshRxahF/ggct3zbihOcDAVl8DfwNsnCsE+ZJc\nmwuuF7x5J+Q51/ArpbUEQjlfAQKBgQDHOnglnw6oNDIbAKwlEwNfJuwOwoolLJLx\nB88Ls8OKSHYRyt141ZHiNYiwoyuWZeu1cmSQB6Ss2igGqqC6/sx+0XnXaB0ngp98\nfUIPYxQqfEWn/7BNwDgstZY1KhgM9XkyqjuYCTJhq3MLap5od77nxNPEOJ+HtbeQ\n3t0JaGdgdwKBgGIRXyxg/Mgv9cDvsTEyrmcV5R5ajsi5T6uoDafTk7S2HaqoVy3e\nXUqdvqHfCa4E8ED6JJYNbo3oeuQIjj4I0cZZtDMaPrUso+gcwEOyS/y8YW0TWZFL\nx/OT2XkbINZRtL+Q4q9fVrruwjRzSVNXQMFSYGONCVfQfex0/l7P4skBAoGAWBdg\nXLIx1uoNZacsdtArY31gTz5xuvI0nuLnB7OauKBFpKRgvTch5DXrlL7xXPT//iDw\nmkbm548mt5vmqghT/5c8GqTsjzXQs0jnVspmdkqwuhHysM5XiF1aZ3OPYtt/lYl0\nBEu8vTcEDX49QNAB15VOVar7zxPocOQ6NBi37Q0CgYEAuMdjiPdjB9g51rzZzthd\n4j92CEv+k+a/mCHRLern/1DAraK9lxYEwYq2iIOx5MM+AZ6IVEuNZpoteQqPSwbR\ntDthjTQ13/InyPV+RhDv6fsCsKydAGiYAu5dB8RiJSge5+opARLS4uCD1NiSHU7D\nv6Ir9SRw6lCVRSeJzIH1gp4=\n-----END PRIVATE KEY-----\n",
    "client_email": "cbbbot@cbbbot-413503.iam.gserviceaccount.com",
    "client_id": "108090732316867851436",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/cbbbot%40cbbbot-413503.iam.gserviceaccount.com",
    "universe_domain": "googleapis.com"
}

# Set up Google Sheets API credentials
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(credentials_info, scope)
client = gspread.authorize(creds)

# Open the Google Sheet by title
spreadsheet_title = "CBBTracking"
spreadsheet = client.open(spreadsheet_title)

# Access the specific worksheet within the spreadsheet
worksheet_title = "Sheet3"
worksheet = spreadsheet.worksheet(worksheet_title)

# Convert filtered_df to a list of dictionaries
new_data = filtered_df[['home_team', 'away_team', 'Predicted_Average_Score', 'point', 'Variance', 'Over_Under', 'Commence_Time_CST', 'commence_time']].to_dict(orient='records')

# Load existing data from the worksheet
existing_data = worksheet.get_all_records()

# Check for duplicate entries based on home team, away team, and commence date
for entry in new_data:
    # Convert commence_time to string for comparison
    entry['Commence_Time_CST'] = str(entry['Commence_Time_CST'])
    entry['commence_time'] = entry['commence_time'].strftime("%Y-%m-%dT%H:%M:%SZ")  # Format with "T" and "Z"

    # Check if the entry already exists
    is_duplicate = any(
        (entry['home_team'] == row['home_team'] and
         entry['away_team'] == row['away_team'] and
         entry['Commence_Time_CST'] == str(row['Commence_Time_CST']) and
         entry['commence_time'] == row['commence_time'])
        for row in existing_data
    )

    # If not a duplicate, append the new entry to the worksheet
    if not is_duplicate:
        worksheet.append_row([entry['home_team'], entry['away_team'], entry['point'], entry['Predicted_Average_Score'],  entry['Variance'], entry['Over_Under'], entry['Commence_Time_CST'], entry['commence_time']])

# Set API endpoint URL for historical outcomes
historical_url = 'https://api.the-odds-api.com/v4/sports/basketball_ncaab/scores/?daysFrom=3&apiKey=00ab27442da4a2b1f8460c5c70d0b3d8'

# Make the request and get the response data for historical outcomes
historical_response = requests.get(historical_url, headers=headers)

if historical_response.status_code == 200:
    historical_data = historical_response.json()
else:
    print(f"Error fetching historical outcomes. Status code: {historical_response.status_code}")

from datetime import datetime

# Iterate through the existing data in the Google Sheet
for i, row in enumerate(existing_data, start=2):  # Start from row 2 assuming headers are in row 1
    home_team = row['home_team']
    away_team = row['away_team']
    commence_time_cst = row['commence_time']

    # Check if final_score and over_under_result are already present
    if 'final_score' not in row or 'over_under_result' not in row:
        # Extract the date part of the commence time from the Google Sheet row
        commence_date_sheet = datetime.strptime(commence_time_cst, "%Y-%m-%dT%H:%M:%SZ").date()

        # Find historical outcome for the corresponding game
        relevant_outcome = None
        for outcome in historical_data:
            # Check if the game is completed
            if outcome.get('completed', False):  # Only proceed if 'completed' is True
                # Extract the date part of the commence time from the API response
                commence_date_api = datetime.strptime(outcome['commence_time'], "%Y-%m-%dT%H:%M:%SZ").date()

                # Compare only the date parts of the commence times
                if (outcome.get('home_team') == home_team and
                    outcome.get('away_team') == away_team and
                    commence_date_sheet == commence_date_api):
                    relevant_outcome = outcome
                    break

        # If a relevant historical outcome is found, update the Google Sheet
        if relevant_outcome:
            # Check if 'scores' key is available and not None
            if 'scores' in relevant_outcome and relevant_outcome['scores'] is not None:
                # Access the scores of home and away teams
                home_score = next((item['score'] for item in relevant_outcome['scores'] if item['name'] == home_team), None)
                away_score = next((item['score'] for item in relevant_outcome['scores'] if item['name'] == away_team), None)

                if home_score is not None and away_score is not None:
                    final_score = int(home_score) + int(away_score)
                    point_entered = int(row['point'])  # Assuming 'point' column is available in the Google Sheet
                    over_under_result = 'Over' if final_score > point_entered else 'Under'

                    # Update the final score and result in the Google Sheet
                    worksheet.update_cell(i, 9, final_score)  # Assuming 'Variance' column is in the 9th position
                    worksheet.update_cell(i, 10, over_under_result)  # Assuming 'Over_Under' column is in the 10th position
                else:
                    print(f"Scores not available for game: {home_team} vs {away_team} at {commence_time_cst}")
            else:
                print(f"No scores available for game: {home_team} vs {away_team} at {commence_time_cst}")
        else:
            print(f"No historical outcome found for game: {home_team} vs {away_team} at {commence_time_cst}")
