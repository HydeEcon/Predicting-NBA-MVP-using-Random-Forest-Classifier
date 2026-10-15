import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report
)

# Clean Data

df = pd.read_csv("NBA_player_stats_1980_2026.csv")

# Returns a list of all columns containing strings/text to be removed

string_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
print(string_cols)
string_df = df.select_dtypes(include=['object', 'string'])
string_df.head()
string_df.info()

df_clean = df.drop(
    columns=['Rk', 'Pos','Team', 
    'stat_source', 'per_game_Awards', 'per_game_stat_source', 
    'per_36_Awards', 'per_36_stat_source', 'per_100_Awards', 
    'per_100_stat_source', 'advanced_Awards', 'advanced_stat_source']
).dropna()

df_clean = df_clean[df_clean['Awards'].str.contains(r'\bMVP\b', regex = True, na = False)] # Only show players in MVP race, keeps data set balanced
df_clean['Awards'] = 0
df_clean['MPPG'] = df_clean['MP'] / df_clean['G']
df_clean = df_clean.drop(columns=['MP'])
df_clean['szn'] = df['Season'].str.split('-').str[0].astype(int)
df_clean = df_clean.drop(columns=['Season'])

threshold_SZN = 2001 # When zone defense was allowed
threshold_GP = 49 # For games played, No MVP ever played less than 49 games
threshold_PTS = 15.5 # Lowest PPG since 1980
threshold_FGA = 10.9 # And so on and so on
threshold_TRB = 3.3
threshold_AST = 1.3
threshold_FG = 0.37
threshold_MPPG = 30.4

df_clean = df_clean[(df_clean['G'] >= threshold_GP) 
                    & (df_clean['PTS'] >= threshold_PTS) 
                    & (df_clean['FGA'] >= threshold_FGA) 
                    & (df_clean['TRB'] >= threshold_TRB) 
                    & (df_clean['AST'] >= threshold_AST) 
                    & (df_clean['MPPG'] >= threshold_MPPG)
                    & (df_clean['szn'] >= threshold_SZN)
                    & (df_clean['FG%'] >= threshold_FG)]

df_clean.info()
df_clean.head()
print(df_clean['szn'].value_counts())

# Random Forest Walk 1, get a feel for things

from sklearn.model_selection import GridSearchCV, train_test_split

X = df_clean.drop(columns=['MVP', 'Player', 'Awards', 'szn'])
y = df_clean['MVP']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17, test_size=0.2)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
rf.score(X_test, y_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

features = pd.DataFrame(rf.feature_importances_, index = X.columns, columns=["X"])
features.sort_values(by='X', ascending=False).head(15)

# Poor results, model defaults to 0, so lets add some hyper parameters and seasonal loops

all_predictions = []
season_results = []

# Test each season starting in 2001
for season in sorted(df_clean['szn'].unique()):
    if season < 2000:
        continue

    # Train only on seasons BEFORE the test season
    train_df = df_clean[df_clean['szn'] < season].copy()

    # Test on the current season
    test_df = df_clean[df_clean['szn'] == season].copy()

    # Skip if there is no historical training data
    if train_df.empty or test_df.empty:
        continue

    # Define features and target
    X_train = train_df.drop(
        columns=['MVP', 'Player', 'Awards', 'szn']
    )
    y_train = train_df['MVP']

    X_test = test_df.drop(
        columns=['MVP', 'Player', 'Awards', 'szn']
    )
    y_test = test_df['MVP']

    # Need both classes in training data
    if y_train.nunique() < 2:
        print(f"Skipping {season}: training data has only one class")
        continue

    # Train model using previous seasons
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=17,
        class_weight='balanced'
    )

    rf.fit(X_train, y_train)

    # Predict all eligible players in the test season
    y_pred = rf.predict(X_test)

    # Store player-level predictions
    season_predictions = pd.DataFrame({
        'Player': test_df['Player'].values,
        'Season': season,
        'Actual MVP': y_test.values,
        'Predicted MVP': y_pred
    })

    season_predictions['Actual Result'] = (
        season_predictions['Actual MVP']
        .map({0: 'Did not win', 1: 'Won MVP'})
    )

    season_predictions['Predicted Result'] = (
        season_predictions['Predicted MVP']
        .map({0: 'Did not predict MVP', 1: 'Predicted MVP'})
    )

    season_predictions['Correct Prediction'] = (
        season_predictions['Actual MVP']
        == season_predictions['Predicted MVP']
    )

    all_predictions.append(season_predictions)

    # Store season-level performance
    season_results.append({
        'Season': season,
        'Training Seasons': f"Before {season}",
        'Players Tested': len(test_df),
        'Accuracy': accuracy_score(y_test, y_pred)
    })

    print(f"Finished season {season}")

# Combine all seasons into one table
if all_predictions:
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    results_df = pd.DataFrame(season_results)

    # Show player predictions, newest season first
    predictions_df = predictions_df.sort_values(
        ['Season', 'Predicted MVP'],
        ascending=[False, False]
    )

    print(predictions_df)
    print(results_df)
else:
    print("No seasons could be evaluated.")

mvp_predictions = predictions_df[
    (predictions_df['Actual MVP'] == 1) |
    (predictions_df['Predicted MVP'] == 1)
]

print(mvp_predictions)

print(classification_report(
    predictions_df['Actual MVP'],
    predictions_df['Predicted MVP'],
    target_names=['Did Not Win MVP', 'Won MVP'],
    zero_division=0
))

mvp_df = pd.DataFrame(mvp_predictions)
mvp_df.to_excel('MVP_Predictions_RFW.xlsx', index=False)
print("MVP predictions saved to MVP_Predictions_RFW.xlsx")