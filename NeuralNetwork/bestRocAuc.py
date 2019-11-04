from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_predict
import pandas as pd


def getRocAucScore(model, dataframe):
    X = dataframe.iloc[:, :-1]
    y = dataframe.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.3)
    y_train_pred = cross_val_predict(model, X_train, y_train, cv=10,)
    y_scores = y_train_pred[:, 1]
    return roc_auc_score(y_train, y_scores)
