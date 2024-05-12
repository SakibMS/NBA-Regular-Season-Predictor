import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from Models import backtest


df = pd.read_csv("nba_games.csv", index_col=0)

# Preparing DataFrame
df = df.sort_values("date")
df = df.reset_index(drop=True)

# Deleting duplicate data
del df["mp.1"]
del df["mp_opp.1"]
del df["index_opp"]

# Given a DataFrame consisting of one team's games record whether that team won their next game

# Replacing target value at instances without a next game
df["target"][pd.isnull(df["target"])] = 2

# Replace True and False with 1 and 0
df["target"] = df["target"].astype(int, errors="ignore")

# Finding null columns
nulls = pd.isnull(df)
nulls = nulls.sum()
nulls = nulls[nulls > 0]

# Deleting null columns
valid_columns = df.columns[~df.columns.isin(nulls.index)]
df = df[valid_columns].copy()

rr = RidgeClassifier(alpha=1)
split = TimeSeriesSplit(n_splits=3)

sfs = SequentialFeatureSelector(rr, n_features_to_select=30, direction="forward", cv=split)

# Scaling values in appropriate columns such that they fall between 0 and 1
removed_columns = ["season", "date", "won", "target", "team", "team_opp"]
selected_columns = df.columns[~df.columns.isin(removed_columns)]

scaler = MinMaxScaler()
df[selected_columns] = scaler.fit_transform(df[selected_columns])

sfs.fit(df[selected_columns], df["target"])

predictors = list(selected_columns[sfs.get_support()])

# The following function trains the model on data of the previous seasons
# requiring there to be at least 2 previous seasons


predictions = backtest(df, rr, predictors)

predictions = predictions[predictions["actual"] != 2]
accuracy_score(predictions["actual"], predictions["prediction"])


