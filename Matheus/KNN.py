import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.utils.validation import column_or_1d

datasets = ["a_affirmative", "a_conditional", "a_doubt_question", "a_emphasis", "a_negative", "a_relative", "a_topics", "a_wh_question", "a_yn_question",
         "b_affirmative", "b_conditional", "b_doubt_question", "b_emphasis", "b_negative", "b_relative", "b_topics", "b_wh_question", "b_yn_question"]

for dataset in datasets:
    X_train = pd.read_csv(dataset + "_X_train.csv")
    X_test = pd.read_csv(dataset + "_X_test.csv")
    y_train = pd.read_csv(dataset + "_y_train.csv")
    y_train = column_or_1d(y_train, warn=False)
    y_test = pd.read_csv(dataset + "_y_test.csv")
    y_test = column_or_1d(y_test, warn=False)
    k_range = range(1, 30)
    k_scores = []
    best_k = 1
    k_score = 0
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, weights = 'distance', 
                                   algorithm = 'ball_tree', leaf_size = 1)
        y_train_proba = cross_val_predict(knn,X_train,y_train,cv=10,method="predict_proba")
        y_scores = y_train_proba[:, 1]
        y_score = roc_auc_score(y_train, y_scores)
        if y_score > k_score:
            best_k = k
            k_score = y_score
    print("Melhores parametros para {0} : k = {1}, score = {2}".format(dataset, best_k, k_score))