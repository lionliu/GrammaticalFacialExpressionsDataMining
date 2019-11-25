import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils.validation import column_or_1d
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score


datasets = ["a_affirmative", "a_conditional", "a_doubt_question", "a_emphasis", "a_negative", "a_relative", "a_topics", "a_wh_question", "a_yn_question",
         "b_affirmative", "b_conditional", "b_doubt_question", "b_emphasis", "b_negative", "b_relative", "b_topics", "b_wh_question", "b_yn_question"]
alphas = {"a_affirmative": 0.0001, "a_conditional": 0.0001, "a_doubt_question": 0.0001, "a_emphasis": 0.0001, "a_negative": 0.0003, "a_relative": 0.0003, "a_topics": 0.0003, "a_wh_question": 0.0003, "a_yn_question": 0.0001,
         "b_affirmative": 0.01, "b_conditional": 0.01, "b_doubt_question": 0.001, "b_emphasis": 0.001, "b_negative": 0.003, "b_relative": 0.003, "b_topics": 0.01, "b_wh_question": 0.0001, "b_yn_question": 0.003}

lr = {"a_affirmative": 0.01, "a_conditional": 0.01, "a_doubt_question": 0.01, "a_emphasis": 0.01, "a_negative": 0.01, "a_relative": 0.01, "a_topics": 0.01, "a_wh_question": 0.01, "a_yn_question": 0.003,
         "b_affirmative": 0.01, "b_conditional": 0.01, "b_doubt_question": 0.01, "b_emphasis": 0.01, "b_negative": 0.01, "b_relative": 0.01, "b_topics": 0.01, "b_wh_question": 0.01, "b_yn_question": 0.01}

estimators = {"a_affirmative": 100, "a_conditional": 25, "a_doubt_question": 100, "a_emphasis": 100, "a_negative": 100, "a_relative": 50, "a_topics": 50, "a_wh_question": 10, "a_yn_question": 25,
         "b_affirmative": 25, "b_conditional": 50, "b_doubt_question": 25, "b_emphasis": 100, "b_negative": 100, "b_relative": 100, "b_topics": 50, "b_wh_question": 100, "b_yn_question": 100}
count = 0
knn = [5, 5, 3, 3, 3, 3, 7, 3, 4, 3, 4, 4, 3, 4, 5, 5, 6, 3]

for dataset in datasets:
    X_train = pd.read_csv(dataset + "_X_train.csv")
    X_test = pd.read_csv(dataset + "_X_test.csv")
    y_train = pd.read_csv(dataset + "_y_train.csv")
    y_train = column_or_1d(y_train, warn=False)
    y_test = pd.read_csv(dataset + "_y_test.csv")
    y_test = column_or_1d(y_test, warn=False)
    model1 = KNeighborsClassifier(n_neighbors=4, weights = 'distance', 
                                   algorithm = 'ball_tree', leaf_size = 1)
    mlp = MLPClassifier(random_state=42, hidden_layer_sizes=(30, 30), max_iter=50, 
                        activation="relu", solver="adam", alpha=alphas[dataset], learning_rate_init=lr[dataset])
    model2 = BaggingClassifier(mlp, n_estimators=100)
    model3 = DecisionTreeClassifier()
    estim = []
    estim.append(("knn", model1))
    estim.append(("bagging", model2))
    estim.append(("decisiontree", model3))
    ensemble = VotingClassifier(estimators, voting='hard')
    results = model_selection.cross_val_score(ensemble, X_train, y_train, cv=10)
    print(results.mean())
    print("Melhor parametro para {0} : k = {1}, score = {2}".format(dataset, results.mean()))
    count += 1
    
    