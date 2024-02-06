  Objective:
The code aims to predict the variance between the betting line and the predicted final scores for NCAA basketball games.

  Data Collection:
It retrieves odds data for NCAA basketball games from an API (OddsData).
Additional information such as team names, commence time, and average scoring statistics are obtained from web scraping using pandas.

  Data Processing:
The team names in the odds data are standardized using a team mapping from an Excel file.
Various dataframes are created and merged to compile necessary information for predictions.

  Prediction:
Linear regression models are trained separately for home and away teams, predicting the final scores.
The predicted scores are combined to calculate an average predicted score for both teams.

  Analysis:
The variance between the predicted and actual points is calculated.
An 'Over_Under' column is added, indicating whether the prediction is over or under the betting line.

  Time Handling:
Current UTC time is obtained and converted to Central Standard Time (CST).
Commence times are also converted to CST for better readability.

  Performance Tracking:
The code uses googlesheets API to export the prediction results to a google sheet for performance tracking purposes. 
It includes a duplicate check so that the google sheet does not become overloaded with duplicate game data
