import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import column_or_1d
import matplotlib.pyplot as plt
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
simplefilter("ignore", category=ConvergenceWarning)
simplefilter("ignore", category=DataConversionWarning)
import numpy as np

datasets = ["a_affirmative", "a_conditional", "a_doubt_question", "a_emphasis", "a_negative", "a_relative", "a_topics", "a_wh_question", "a_yn_question",
            "b_affirmative", "b_conditional", "b_doubt_question", "b_emphasis", "b_negative", "b_relative", "b_topics", "b_wh_question", "b_yn_question"]
        
# datasets = ["a_affirmative"]

alpha = [0.0001, 0.0003, 0.001, 0.003, 0.01]
learning_rate_init = [0.001, 0.003, 0.01, 0.03]

combinations = range(0, 20) # N de combinacoes

plt.title("ROC AUC Score Combinations for MLP")

bestsParams = {}

for data in datasets:
    X_train = pd.read_csv("../SplitPreprocessedData/" +
                        data + "_X_train.csv")
    X_test = pd.read_csv("../SplitPreprocessedData/" +
                        data + "_X_test.csv")
    y_train = pd.read_csv("../SplitPreprocessedData/" +
                        data + "_y_train.csv")
    y_train = column_or_1d(y_train, warn=True)
    y_test = pd.read_csv("../SplitPreprocessedData/" +
                        data + "_y_test.csv")
    y_test = column_or_1d(y_test, warn=True)

    roc = []
    # bestRocAucScore = 0
    for lri in learning_rate_init:
        for a in alpha:
            mlp = MLPClassifier(random_state=42, hidden_layer_sizes=(30, 30, 30), max_iter=50,
                                activation="relu", solver="adam", alpha=a, learning_rate_init=lri)
            y_train_proba = cross_val_predict(
                mlp, X_train, y_train, cv=10, method="predict_proba")
            y_scores = y_train_proba[:, 1]
            score = roc_auc_score(y_train, y_scores)
            # if(score > bestRocAucScore):
            #         bestsParams[data] = {
            #             "alpha": a,
            #             "learning_rate_init": lri,
            #             "RocAucScore": score
            #         }
            #         bestRocAucScore = score
            roc.append(score)
    plt.plot(combinations, roc, label=data)

plt.legend()
plt.ylabel("ROC AUC")
plt.xlabel("Combination")
plt.xticks(list(combinations))
plt.legend()

plt.show()

# for i in bestsParams:
#     print(i, " :", bestsParams[i])
