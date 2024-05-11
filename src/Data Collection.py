import pandas as pd

df = pd.read_csv("nba_games.csv", index_col=0)

# Preparing DataFrame
df = df.sort_values("date")
df = df.reset_index(drop=True)

# Deleting duplicate data
del df["mp.1"]
del df["mp_opp.1"]
del df["index_opp"]

# Given a DataFrame consisting of one team's games record whether that team won their next game
def add_target(team):
    team["target"] = team["won"].shift(-1)
    return team

df = df.groupby("team", group_keys=False).apply(add_target)
df[df["team"] == "WAS"]

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

