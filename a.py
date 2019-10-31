import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from scipy import stats
from support import getDFPointsAngles
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def genCSV(name):
    dataset = pd.read_csv("./dataset/" + name +"_datapoints.csv", sep=' ')
    targets = pd.read_csv("./dataset/"+ name + "_targets.csv")
    dataset = pd.concat([dataset, targets], axis=1)
    dataset.drop([dataset.columns[0]], axis=1, inplace=True)

    dataset2 = getDFPointsAngles(dataset, targets)

    z = np.abs(stats.zscore(dataset2))
    dataset2 = dataset2[(z < 2.5).all(axis=1)]

    X = dataset2.iloc[:, :-1]
    y = dataset2.iloc[:, -1]

    XScaled = MinMaxScaler().fit_transform(X)
    normDataset = pd.DataFrame(XScaled, columns=['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9',
                                                'd10', 'd11', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7'])
    normDataset['target'] = dataset['target']

    normDataset.to_csv("./PreprocessedDataset/" + name + ".csv", index=False, sep=" ")


names = ["a_affirmative", "a_conditional", "a_doubt_question", "a_emphasis", "a_negative", "a_relative", "a_topics", "a_wh_question", "a_yn_question",
         "b_affirmative", "b_conditional", "b_doubt_question", "b_emphasis", "b_negative", "b_relative", "b_topics", "b_wh_question", "b_yn_question", ]

for i in names:
    genCSV(i)
