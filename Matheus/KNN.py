import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.utils.validation import column_or_1d
from time import perf_counter
from warnings import simplefilter
from sklearn.exceptions import DataConversionWarning
simplefilter("ignore", category=DataConversionWarning)

datasets = ["a_affirmative", "a_conditional", "a_doubt_question", "a_emphasis", "a_negative", "a_relative", "a_topics", "a_wh_question", "a_yn_question",
         "b_affirmative", "b_conditional", "b_doubt_question", "b_emphasis", "b_negative", "b_relative", "b_topics", "b_wh_question", "b_yn_question"]

k = {"a_affirmative": 5, "a_conditional": 5, "a_doubt_question": 3, "a_emphasis": 3, "a_negative": 3, "a_relative": 3, "a_topics": 7, "a_wh_question": 3, "a_yn_question": 4,
      "b_affirmative": 3, "b_conditional": 4, "b_doubt_question": 4, "b_emphasis": 3, "b_negative": 4, "b_relative": 5, "b_topics": 5, "b_wh_question": 6, "b_yn_question": 3}

ks = [3, 4, 5, 6, 7]

combinations = range(0, 5)  # N de combinacoes

plt.title("ROC AUC Score Combinations for KNN")

for dataset in datasets:
      X_train = pd.read_csv("../SplitPreprocessedData/" +
                              dataset + "_X_train.csv")
      X_test = pd.read_csv("../SplitPreprocessedData/" +
                        dataset + "_X_test.csv")
      y_train = pd.read_csv("../SplitPreprocessedData/" +
                        dataset + "_y_train.csv")
      y_train = column_or_1d(y_train, warn=True)
      y_test = pd.read_csv("../SplitPreprocessedData/" +
                        dataset + "_y_test.csv")
      y_test = column_or_1d(y_test, warn=True)
      roc = []

      for k in ks:
            knn = KNeighborsClassifier(n_neighbors=k, weights = 'distance', 
                                          algorithm = 'ball_tree', leaf_size = 1)
            y_train_proba = cross_val_predict(
            knn, X_train, y_train, cv=10, method="predict_proba")
            y_scores = y_train_proba[:, 1]
            score = roc_auc_score(y_train, y_scores)
            roc.append(score)

      plt.plot(combinations, roc, label=dataset)


plt.legend()
plt.ylabel("ROC AUC")
plt.xlabel("Combination")
plt.xticks(list(combinations))
plt.legend()

plt.show()
