import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from scipy import stats
from support import getDFPointsAngles
from sklearn.preprocessing import MinMaxScaler, StandardScaler

dataset = pd.read_csv("./dataset/a_affirmative_datapoints.csv", sep=' ')
targets = pd.read_csv("./dataset/a_affirmative_targets.csv")
dataset = pd.concat([dataset, targets], axis=1)
dataset.drop([dataset.columns[0]], axis=1, inplace=True)

dataset2 = getDFPointsAngles(dataset, targets)
print(dataset2.describe())
dataset2.plot(kind='box', subplots=True, layout=(
    4, 5), sharex=False, sharey=False)
plt.show()

z = np.abs(stats.zscore(dataset2))

dataset2 = dataset2[(z < 2.5).all(axis=1)]

dataset2.plot(kind='box', subplots=True, layout=(
    4, 5), sharex=False, sharey=False)
plt.show()

X = dataset2.iloc[:, :-1]
y = dataset2.iloc[:, -1]

XScaled = MinMaxScaler().fit_transform(X)
normDataset = pd.DataFrame(XScaled, columns=['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9',
                                             'd10', 'd11', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7'])
normDataset['target'] = dataset['target']
print(normDataset.describe())

