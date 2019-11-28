import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ["a_affirm", "a_cond", "a_doubt", "a_emph", "a_neg", "a_relat", "a_topics", "a_wh", "a_yn",
          "b_affirm", "b_cond", "b_doubt", "b_emph", "b_neg", "b_relat", "b_topics", "b_wh", "b_yn"]
# MLPRoc = [0.666,0.704,0.667,0.595,0.643,0.733,0.686,0.671,0.620,0.699,0.752,0.754,0.773,0.768,0.788,0.700,0.827,0.778]
# BagRoc = [0.696,0.703,0.691,0.616,0.665,0.735,0.720,0.676,0.644,0.696,0.765,0.796,0.777,0.736,0.786,0.729,0.830,0.785]

MLPRoc = [0.698,0.911,0.880,0.628,0.653,0.902,0.755,0.712,0.839,0.618,0.727,0.766,0.781,0.762,0.742,0.864,0.867,0.782,]
BagRoc = [0.738,0.918,0.909,0.770,0.738,0.893,0.819,0.740,0.890,0.679,0.789,0.841,0.836,0.800,0.781,0.887,0.913,0.802,]

x = np.arange(len(labels))  
width = 0.45  

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, MLPRoc, width, label='MLP')
rects2 = ax.bar(x + width/2, BagRoc, width, label='Ensemble')

ax.set_ylabel('ROC AUC Scores')
ax.set_title("Original Dataset Comparison")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()
