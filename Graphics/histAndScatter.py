import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import sys
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
sys.path.insert(1, "../")
from support import getDFPointsAngles

data = "b_wh_question"

dataset = pd.read_csv("../dataset/" + data + "_datapoints.csv", sep=' ')
targets = pd.read_csv("../dataset/" + data + "_targets.csv")
dataset = pd.concat([dataset, targets], axis=1)
dataset.drop([dataset.columns[0]], axis=1, inplace=True)

# histograma para as primeiras 12 features

# plt.title("Histogram for 12 features of the b_wh_question dataset")
# dataset.iloc[:, :12].hist()
# scatter_matrix(dataset.iloc[:, :6])
# plt.show()


dataset2 = getDFPointsAngles(dataset, targets)

# dataset2.iloc[:, :-1].hist()
# scatter_matrix(dataset2.iloc[:, :-1])
# plt.show()

z = np.abs(stats.zscore(dataset2))
dataset2 = dataset2[(z < 2.5).all(axis=1)]

X = dataset2.iloc[:, :-1]
y = dataset2.iloc[:, -1]

XScaled = MinMaxScaler().fit_transform(X)
normDataset = pd.DataFrame(XScaled, columns=['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9',
                                             'd10', 'd11', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7'])
normDataset.hist()
plt.show()
normDataset['target'] = dataset['target']
