import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statistics import mean


treeDefault = [0.92,0.97,0.95,0.96,0.92,0.97,0.96,0.94,0.95,0.91,0.89,0.93,0.94,0.94,0.96,0.93,0.94,0.92,]

knnDefault = [0.85,0.95,0.92,0.88,0.89,0.94,0.89,0.91,0.93,0.83,0.85,0.83,0.84,0.83,0.81,0.87,0.92,0.81,]

mlpDefault = [0.61,0.84,0.87,0.78,0.49,0.76,0.76,0.46,0.76,0.55,0.77,0.71,0.63,0.68,0.71,0.83,0.79,0.61,]

forestDefault = [0.89,0.98,0.92,0.97,0.89,0.97,0.96,0.93,0.94,0.87,0.89,0.92,0.93,0.94,0.97,0.92,0.94,0.90,]

baggingDefault = [0.72,0.87,0.85,0.75,0.71,0.85,0.79,0.68,0.84,0.67,0.69,0.752,0.761,0.76,0.69,0.84,0.86,0.74,]

votingDefault = [0.83,0.95,0.90,0.91,0.88,0.95,0.91,0.90,0.92,0.84,0.87,0.86,0.89,0.89,0.91,0.89,0.90,0.83,]

scoresDefault = [treeDefault, knnDefault, mlpDefault, forestDefault, baggingDefault, votingDefault]
scoresMean = []

for s in scoresDefault:
    scoresMean.append(float(format(mean(s), '.3f')))


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        plt.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

x = np.arange(7)


plt.figure()
plt.title("Default dataset mean scores")
plt.xlabel('Classifiers')
plt.ylabel('Mean Scores')
# rect = plt.bar(x, scoresMean)
# autolabel(rect)

labels = ['tree', 'KNN', 'MLP', 'RF', 'Bagging', 'Voting']
box = plt.boxplot(scoresDefault, labels=labels, patch_artist=True)
# plt.xticks(x, ('', 'tree', 'KNN', 'MLP', 'RF', 'Bagging', 'Voting'))
plt.show()