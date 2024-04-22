import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report

# Load datasets into dataframes
kenpom = pd.read_csv('data/KenPom Barttorvik.csv')
preseason = pd.read_csv('data/Preseason Votes.csv')
team = pd.read_csv('data/Team Results.csv')
matchups = pd.read_csv('data/Tournament Matchups.csv')

# Merge datasets
first_merged_df = pd.merge(kenpom, preseason, on=['TEAM', 'YEAR'], how='left')
second_merged_df = pd.merge(first_merged_df, team, on='TEAM', how='left')
final_merged_df = pd.merge(second_merged_df, matchups, on=['TEAM', 'YEAR'], how='right')

# Extract features and target variable from the merged dataframe
X = final_merged_df.drop(columns=['TEAM', 'YEAR', 'WIN%_y'])  # Features
y = final_merged_df['WIN%_y']  # Target variable

# Replace missing values with 0
X.fillna(0, inplace=True)
y.fillna(0, inplace=True)

# Convert categorical variables into dummy/indicator variables
X = pd.get_dummies(X)


# Adding a constant column for the intercept term in the regression model
X = sm.add_constant(X)

# Convert data type to numeric
X = X.apply(pd.to_numeric, errors='coerce')

# Keep only numeric columns in the feature set
X = X.select_dtypes(include=[np.number])

# Adding a constant column again after dropping non-numeric columns
X = sm.add_constant(X)

# Perform stepwise regression

model = sm.OLS(y, X)
result = model.fit()
selected_features = result.summary().tables[1]

print(selected_features)

# Create a DataFrame for selected features
new_features = pd.DataFrame(selected_features.data[1:], columns=selected_features.data[0])

# Select the updated features based on significant features
X_updated = X[['TEAM ID_x', 'ROUND_x', 'K TEMPO RANK', 'BARTHAG', 'GAMES_x', 'W_x', 'WIN%_x', 
               'BADJ EM', 'BADJ D', 'AVG HGT', 'EFF HGT', 'WAB', 'ELITE SOS', 'SEED_y', 
               'ROUND_y', 'AP VOTES', 'AP RANK', 'R64', 'R32', 'S16', 'F4', 'F2', 'CHAMP']]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_updated, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Predict win percentages for each team
team1_features = X_test_scaled[0]  # Features of team 1 from the test set
team2_features = X_test_scaled[1]  # Features of team 2 from the test set

# Predict win percentages for each team
team1_win_percentage = linear_model.predict([team1_features])[0]
team2_win_percentage = linear_model.predict([team2_features])[0]

# Get team names for the test set rows
team1_name = final_merged_df.iloc[X_test.index[0]]['TEAM']
team2_name = final_merged_df.iloc[X_test.index[1]]['TEAM']

# Get actual win percentages for the two teams
team1_actual_win_percentage = y_test.iloc[0]
team2_actual_win_percentage = y_test.iloc[1]

# Print the probability of each team winning
print("Predicted win percentage for", team1_name, ":", team1_win_percentage)
print("Actual win percentage for", team1_name, ":", team1_actual_win_percentage)

print("Predicted win percentage for", team2_name, ":", team2_win_percentage)
print("Actual win percentage for", team2_name, ":", team2_actual_win_percentage)

#test metrics of the model
y_train_pred = linear_model.predict(X_train_scaled)
y_test_pred = linear_model.predict(X_test_scaled)

train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print('Train RMSE:', train_rmse)
print('Test RMSE:', test_rmse)

print('Train R^2:', train_r2)
print('Test R^2:', test_r2)




