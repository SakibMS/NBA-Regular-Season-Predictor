import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Given a DataFrame consisting of one team's games record whether that team won their next game
def add_target(team):
    team["target"] = team["won"].shift(-1)
    return team

df = df.groupby("team", group_keys=False).apply(add_target)
df[df["team"] == "WAS"]

# The following function trains the model on data of the previous seasons
# requiring there to be at least 2 previous seasons
def backtest(data, model, predictors, start=2, step=1):
    all_predictions = []
    seasons = sorted(data["season"].unique())

    for i in range(start, len(seasons), step):
        season = seasons[i]

        train = data[data["season"] < season]
        test = data[data["season"] == season]

        model.fit(train[predictors], train["target"])

        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)

        combined = pd.concat([test["target"], preds], axis=1)
        combined.columns = ["actual", "prediction"]

        all_predictions.append(combined)

    return pd.concat(all_predictions)