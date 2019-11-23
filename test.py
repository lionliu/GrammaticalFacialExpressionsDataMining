import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from scipy import stats
from support import getDFPointsAngles
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

name = "a_affirmative"

dataset = pd.read_csv("./dataset/" + name + "_datapoints.csv", sep=' ')
targets = pd.read_csv("./dataset/" + name + "_targets.csv")
dataset = pd.concat([dataset, targets], axis=1)
dataset.drop([dataset.columns[0]], axis=1, inplace=True)

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X_train, X_test, y_test, y_train = train_test_split(X, y)
pd.DataFrame(X_train).to_csv("./SplitData/" +
                             name + "_X_train.csv", index=False)
pd.DataFrame(X_test).to_csv("./SplitData/" +
                            name + "_X_test.csv", index=False)
pd.DataFrame(y_train).to_csv("./SplitData/" +
                             name + "_y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("./SplitData/" +
                            name + "_y_test.csv", index=False)

