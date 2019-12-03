import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ["a_affirm", "a_cond", "a_doubt", "a_emph", "a_neg", "a_relat", "a_topics", "a_wh", "a_yn",
          "b_affirm", "b_cond", "b_doubt", "b_emph", "b_neg", "b_relat", "b_topics", "b_wh", "b_yn"]

MLPRoc = [0.7332,0.6878,0.5718,0.6144,0.6915,0.6643,0.6473,0.6318,0.7141,0.7595,0.7658,0.7811,0.7074,0.7325,0.8059,0.6770,0.8151,0.7234,]

BagRoc = [0.7802,0.7332,0.6248,0.6687,0.7044,0.7105,0.7673,0.6482,0.7432,0.7744,0.8202,0.8274,0.7621,0.8080,0.8524,0.7710,0.8498,0.7652,]

# MLPRoc = [0.698,0.911,0.880,0.628,0.653,0.902,0.755,0.712,0.839,0.618,0.727,0.766,0.781,0.762,0.742,0.864,0.867,0.782,]
# BagRoc = [0.738,0.918,0.909,0.770,0.738,0.893,0.819,0.740,0.890,0.679,0.789,0.841,0.836,0.800,0.781,0.887,0.913,0.802,]

x = np.arange(len(labels))  
width = 0.45  

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, MLPRoc, width, label='MLP')
rects2 = ax.bar(x + width/2, BagRoc, width, label='Ensemble')

ax.set_ylabel('ROC AUC Scores')
ax.set_title("Preprocessed Dataset Comparison")
# ax.set_title("Original Dataset Comparison")
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
