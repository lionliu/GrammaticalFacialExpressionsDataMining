import pandas as pd
from time import perf_counter
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils.validation import column_or_1d
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

datasets = ["a_affirmative", "a_conditional", "a_doubt_question", "a_emphasis", "a_negative", "a_relative", "a_topics", "a_wh_question", "a_yn_question",
            "b_affirmative", "b_conditional", "b_doubt_question", "b_emphasis", "b_negative", "b_relative", "b_topics", "b_wh_question", "b_yn_question"]
alphas = [0.0001, 0.0001, 0.0001, 0.0001, 0.0003, 0.0003, 0.0003, 0.0003, 0.0001,
           0.01,  0.01, 0.001, 0.001, 0.003, 0.003, 0.01, 0.0001, 0.003]

lr = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.003,
      0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

estimators = [100, 25, 100, 100, 100, 50, 50, 10,25,
              25, 50, 25, 100, 100, 100, 50, 100, 100]
count = 0
knn = [5, 5, 3, 3, 3, 3, 7, 3, 4, 3, 4, 4, 3, 4, 5, 5, 6, 3]
depth = [20,10,30,30,10,50,10,None,20,20,20,None,None,20,50,20,30,30]
features = [4,18,'log2','log2','auto','auto','auto','auto','auto','log2','sqrt',8,4,'log2',None,'log2',14,8]
min_samples_split = [12,12,20,20,12,20,20,12,20,20,20,20,12,20,20,20,20,20]
criterion = ["gini", "entropy","entropy","entropy","gini","gini","gini","entropy","gini","entropy","gini","entropy","gini","entropy","entropy","entropy","gini","gini"]

for count, dataset in enumerate(datasets):
    X_train = pd.read_csv("../SplitData/{}_X_train.csv".format(dataset))
    X_test = pd.read_csv("../SplitData/{}_X_test.csv".format(dataset))
    y_train = pd.read_csv("../SplitData/{}_y_train.csv".format(dataset))
    y_train = column_or_1d(y_train, warn=False)
    y_test = pd.read_csv("../SplitData/{}_y_test.csv".format(dataset))
    y_test = column_or_1d(y_test, warn=False)
    
    seconds = perf_counter()
    model1 = KNeighborsClassifier(n_neighbors=knn[count], weights='distance',
                                  algorithm='ball_tree', leaf_size=1)
    #alpha_var = alphas[dataset]
    #learningrate_var = lr[dataset]
    mlp = MLPClassifier(random_state=42, hidden_layer_sizes=(30, 30), max_iter=50,
                        activation="relu", solver="adam", alpha=alphas[count],
                        learning_rate_init=lr[count])
    #estimators_var = estimators[count]
    model2 = BaggingClassifier(mlp, n_estimators=estimators[count])
    #max_depth_var = depth[count]
    #max_features_var = features[count]
    #min_samples_split_var = min_samples_split[count]
    #criterion_var = criterion
    model3 = DecisionTreeClassifier(criterion=criterion[count],
                                    min_samples_split=min_samples_split[count],
                                    max_features=features[count], max_depth=depth[count])
    estim = []
    estim.append(("knn", model1))
    estim.append(("bagging", model2))
    estim.append(("decisiontree", model3))
    ensemble = VotingClassifier(estim, voting='soft')
    ensemble.fit(X_train, y_train)
    Y_test_prediction = ensemble.predict(X_test)

    print(perf_counter() - seconds)

    # with open("Voting/{}_SCORE.txt".format(dataset), "w+") as wp:
    #     wp.write("Clasification report:\n {}".format(classification_report(y_test, Y_test_prediction)))
    #     wp.write('\n')
    #     wp.write("Confussion matrix:\n {}".format(confusion_matrix(y_test, Y_test_prediction)))
    #     wp.write('\n')
    #     wp.write("Train Score: {}".format(ensemble.score(X_train, y_train)))
    #     wp.write('\n')
    #     wp.write("Test Score: {}".format(ensemble.score(X_test, y_test)))
        
    # print(ensemble.score(X_test, y_test))

    #y_train_proba_mlp = cross_val_predict(
    #    mlp, X_train, y_train, cv=10, method="predict_proba")
    #y_scores_mlp = y_train_proba_mlp[:, 1]
    #MLPScore = roc_auc_score(y_train, y_scores_mlp)
    #parametro =
    #print("Melhor parametro para {0} : param = {1}, score = {2}"
    #      .format(dataset, "arrumando ainda aaa",  MLPScore))
