import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statistics import mean


treePrep = [0.75,0.79,0.70,0.79,0.71,0.78,0.80,0.70,0.73,0.83,0.83,0.84,0.82,0.84,0.85,0.85,0.82,0.81,]

knnPrep = [0.77,0.81,0.82,0.83,0.73,0.84,0.84,0.75,0.78,0.78,0.86,0.88,0.81,0.88,0.93,0.91,0.87,0.89,]

mlpPrep = [0.68,0.72,0.63,0.79,0.64,0.75,0.78,0.65,0.61,0.67,0.7,0.66,0.65,0.66,0.78,0.78,0.80,0.73,]

forestPrep = [0.78,0.78,0.78,0.86,0.74,0.78,0.82,0.69,0.70,0.80,0.82,0.85,0.81,0.81,0.90,0.90,0.82,0.87,]

baggingPrep = [0.72,0.71,0.69,0.82,0.64,0.73,0.79,0.58,0.64,0.67,0.7,0.66,0.67,0.66,0.81,0.78,0.77,0.755,]

votingPrep = [0.72,0.77,0.78,0.85,0.68,0.81,0.80,0.68,0.70,0.77,0.82,0.85,0.82,0.82,0.90,0.89,0.85,0.87,]

scoresPrep = [treePrep, knnPrep, mlpPrep, forestPrep, baggingPrep, votingPrep]
scoresMean = []

for s in scoresPrep:
    scoresMean.append(float(format(mean(s), '.3f')))


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        plt.annotate('{}'.format(height, '.3f'),
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')
                     
x = np.arange(6)


plt.figure()
plt.xlabel('Classifiers')
plt.ylabel('Mean Scores') 
plt.title("Preprocessed dataset mean scores")
rect = plt.bar(x, scoresMean)
autolabel(rect)
plt.bar(x, scoresMean)
plt.xticks(x, ('tree', 'KNN', 'MLP', 'RF', 'Bagging', 'Voting'))
plt.show()

