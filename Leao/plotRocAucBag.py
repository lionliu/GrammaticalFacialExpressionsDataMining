import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import column_or_1d
import matplotlib.pyplot as plt
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
simplefilter("ignore", category=ConvergenceWarning)
simplefilter("ignore", category=DataConversionWarning)

datasets = ["a_affirmative", "a_conditional", "a_doubt_question", "a_emphasis", "a_negative", "a_relative", "a_topics", "a_wh_question", "a_yn_question",
            "b_affirmative", "b_conditional", "b_doubt_question", "b_emphasis", "b_negative", "b_relative", "b_topics", "b_wh_question", "b_yn_question"]

# datasets = ["a_affirmative"]


params = {'a_affirmative': {'alpha': 0.0001, 'learning_rate_init': 0.01, 'RocAucScore': 0.7322321234514235},
          'a_conditional': {'alpha': 0.001, 'learning_rate_init': 0.01, 'RocAucScore': 0.6878647342995169},
          'a_doubt_question': {'alpha': 0.0001, 'learning_rate_init': 0.003, 'RocAucScore': 0.5718023939242163},
          'a_emphasis': {'alpha': 0.01, 'learning_rate_init': 0.01, 'RocAucScore': 0.6144836486716828},
          'a_negative': {'alpha': 0.001, 'learning_rate_init': 0.01, 'RocAucScore': 0.6915968192563937},
          'a_relative': {'alpha': 0.0001, 'learning_rate_init': 0.01, 'RocAucScore': 0.6643406140531629},
          'a_topics': {'alpha': 0.0003, 'learning_rate_init': 0.01, 'RocAucScore': 0.6473959455651511},
          'a_wh_question': {'alpha': 0.001, 'learning_rate_init': 0.003, 'RocAucScore': 0.6318664763325446},
          'a_yn_question': {'alpha': 0.0003, 'learning_rate_init': 0.01, 'RocAucScore': 0.7141722707760443},
          'b_affirmative': {'alpha': 0.003, 'learning_rate_init': 0.01, 'RocAucScore': 0.7595775538857945},
          'b_conditional': {'alpha': 0.01, 'learning_rate_init': 0.01, 'RocAucScore': 0.7658497968334034},
          'b_doubt_question': {'alpha': 0.001, 'learning_rate_init': 0.01, 'RocAucScore': 0.7811996118516573},
          'b_emphasis': {'alpha': 0.0001, 'learning_rate_init': 0.003, 'RocAucScore': 0.7074909383809908},
          'b_negative': {'alpha': 0.0003, 'learning_rate_init': 0.01, 'RocAucScore': 0.7372542293019431},
          'b_relative': {'alpha': 0.0001, 'learning_rate_init': 0.01, 'RocAucScore': 0.8059416272414742},
          'b_topics': {'alpha': 0.001, 'learning_rate_init': 0.01, 'RocAucScore': 0.6770052961001445},
          'b_wh_question': {'alpha': 0.0001, 'learning_rate_init': 0.01, 'RocAucScore': 0.8151934951139722},
          'b_yn_question': {'alpha': 0.01, 'learning_rate_init': 0.003, 'RocAucScore': 0.7234257738267788}}

n_estimators = [10, 25, 50, 100]
combinations = range(0, 4)  # N de combinacoes
bestsParams = {}
plt.title("ROC AUC Score Combinations for Bagging")

for data in datasets:
    X_train = pd.read_csv("../SplitPreprocessedData/" +
                          data + "_X_train.csv")
    X_test = pd.read_csv("../SplitPreprocessedData/" +
                         data + "_X_test.csv")
    y_train = pd.read_csv("../SplitPreprocessedData/" +
                          data + "_y_train.csv")
    y_train = column_or_1d(y_train, warn=False)
    y_test = pd.read_csv("../SplitPreprocessedData/" +
                         data + "_y_test.csv")
    y_test = column_or_1d(y_test, warn=False)

    roc = []
    bestRocAucScore = 0

    # print(params[data]['alpha'])
    # print(params[data]['learning_rate_init'])

    for est in n_estimators:
        mlp = MLPClassifier(random_state=42, hidden_layer_sizes=(30, 30, 30), max_iter=50,
                            activation="relu", solver="adam", alpha=params[data]['alpha'], learning_rate_init=params[data]['learning_rate_init'])
        bag = BaggingClassifier(mlp, n_estimators=est)
        y_train_proba = cross_val_predict(
            bag, X_train, y_train, cv=10, method="predict_proba")
        y_scores = y_train_proba[:, 1]
        score = roc_auc_score(y_train, y_scores)
        # print(est)
        # print(s)
        roc.append(score)
        # if(score > bestRocAucScore):
        #     bestsParams[data] = {
        #         "estimators": est,
        #         "RocAucScore": score
        #     }
        #     bestRocAucScore = score
    plt.plot(combinations, roc, label=data)

plt.legend()
plt.ylabel("ROC AUC")
plt.xlabel("Combination")
plt.xticks(list(combinations))
plt.legend()

plt.show()


for i in bestsParams:
    print(i, " :", bestsParams[i])
