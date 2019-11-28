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
from  warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

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
depth = [7,10,None, 8, 10, None, 10, 8, 8, 10, 10, None, None, None, 9, None, 7]
features = [8, None, 4, 6, 9, 6, 10, 8, 2, 6, 9,6,8,12,8,8,12]
min_samples_split = [9,9,10,3,6,9,7,9,3,4,4,10,9,9,5,10,3]
criterion = ["gini", "entropy", "gini", "entropy", "gini", "gini","entropy","gini", "entropy","entropy","entropy","entropy","entropy","entropy", "gini", "gini","entropy"] 

for dataset in datasets:
    X_train = pd.read_csv(dataset + "_X_train.csv")
    X_test = pd.read_csv(dataset + "_X_test.csv")
    y_train = pd.read_csv(dataset + "_y_train.csv")
    y_train = column_or_1d(y_train, warn=False)
    y_test = pd.read_csv(dataset + "_y_test.csv")
    y_test = column_or_1d(y_test, warn=False)
    knn_var = knn[count]
    model1 = KNeighborsClassifier(n_neighbors=knn_var, weights = 'distance', 
                                   algorithm = 'ball_tree', leaf_size = 1)
    alpha_var = alphas[dataset]
    learningrate_var = lr[dataset]
    mlp = MLPClassifier(random_state=42, hidden_layer_sizes=(30, 30), max_iter=50, 
                        activation="relu", solver="adam", alpha=alpha_var, 
                        learning_rate_init=learningrate_var)
    estimators_var = estimators[dataset]
    model2 = BaggingClassifier(mlp, n_estimators=estimators_var)
    max_depth_var = depth[count]
    max_features_var = features[count]
    min_samples_split_var = min_samples_split[count]
    criterion_var = criterion[count]
    model3 = DecisionTreeClassifier(criterion = criterion_var, 
                                    min_samples_split = min_samples_split_var,
                                    max_features = max_features_var, max_depth = max_depth_var)
    ensemble = VotingClassifier(estimators=[("knn",model1),("bagging",model2),("decisiontree",model3)], voting='soft')
    y_train_proba = cross_val_predict(
            ensemble, X_train, y_train, cv=10, method="predict_proba")
    y_scores = y_train_proba[:, 1]
    Score = roc_auc_score(y_train, y_scores)
    print("Melhor parametro para {0} : score = {1}"
          .format(dataset,  Score))
    count += 1
    
    