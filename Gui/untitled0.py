import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.validation import column_or_1d
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn import tree

datasets = ["a_affirmative", "a_conditional", "a_doubt_question", "a_emphasis", "a_negative", "a_relative", "a_topics", "a_wh_question", "a_yn_question",
             "b_affirmative", "b_conditional", "b_doubt_question", "b_emphasis", "b_negative", "b_relative", "b_topics", "b_wh_question"]

bestsParams = {}

max_depth= [1,2,3,4,5,6,7,8,9,10, None]
max_features= [1,2,3,4,5,6,7,8,9,10,11,12, 'auto', 'sqrt', 'log2', None]
min_samples_split= [2, 3,4,5,6,7,8,9, 10]
criterion= ["gini", "entropy"]

writer= open("decisiontreePREparam5.txt", "w+")

for d in datasets:
    
    X_train = pd.read_csv("../SplitPreprocessedData/{}_X_train.csv".format(d))
    X_test = pd.read_csv("../SplitPreprocessedData/{}_X_test.csv".format(d))
    y_train = pd.read_csv("../SplitPreprocessedData/{}_y_train.csv".format(d))
    y_train = column_or_1d(y_train, warn=True)
    y_test = pd.read_csv("../SplitPreprocessedData/{}_y_test.csv".format(d))
    y_test = column_or_1d(y_test, warn=True)
    max_score=0
    for c in criterion:
        for mins in min_samples_split:
            for maxf in max_features:
                for maxd in max_depth:
                    clf = tree.DecisionTreeClassifier(criterion=c,min_samples_split=mins,max_features=maxf,max_depth=maxd)
                    clf.fit(X_train, y_train)
                    if (clf.score(X_train, y_train) + clf.score(X_test, y_test)) > max_score:
                        bestsParams[d]={
                                "criterion":c,
                                "max_depth":maxd,
                                "max_features":maxf,
                                "min_samples_split":mins,
                                "train_score":clf.score(X_train, y_train),
                                "test_score":clf.score(X_test, y_test)
                                }
                        max_score = (clf.score(X_train, y_train) + clf.score(X_test, y_test))
for i in bestsParams:
    writer.write(str(i))
    writer.write('\n')
    writer.write(str(bestsParams[i]))
    writer.write('\n')
    
writer.close()