import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.validation import column_or_1d
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

datasets = ["a_affirmative", "a_conditional", "a_doubt_question", "a_emphasis", "a_negative", "a_relative", "a_topics", "a_wh_question", "a_yn_question",
             "b_affirmative", "b_conditional", "b_doubt_question", "b_emphasis", "b_negative", "b_relative", "b_topics", "b_wh_question", "b_yn_question"]

bestsParams = {}

maxd= [30,5,5,5,5,5,None,5,5,5,5,5,100,50,None,5,5,None]
maxf= [60,20,'sqrt',60,40,80,60,'auto',8,60,14,80,20,60,'auto','sqrt',60,'sqrt']
mins= [20,12,20,20,20,20,20,20,2,20,6,12,20,20,12,20,20,20]
c= ["entropy","entropy","entropy","entropy","entropy","entropy","entropy","entropy","entropy","entropy","entropy","entropy","entropy","gini","entropy","entropy","entropy","entropy"]
e=[100,50,50,20,100,20,100,200,200,100,200,200,100,100,200,200,50,200]

#writer= open("Forest/forestSCORE.txt", "w+")
#scorelist=[]

for cnt, d in enumerate(datasets):
    
    X_train = pd.read_csv("../SplitData/{}_X_train.csv".format(d))
    X_test = pd.read_csv("../SplitData/{}_X_test.csv".format(d))
    y_train = pd.read_csv("../SplitData/{}_y_train.csv".format(d))
    y_train = column_or_1d(y_train, warn=True)
    y_test = pd.read_csv("../SplitData/{}_y_test.csv".format(d))
    y_test = column_or_1d(y_test, warn=True)
    
    clf = RandomForestClassifier(n_estimators=e[cnt], criterion=c[cnt],min_samples_split=mins[cnt],max_features=maxf[cnt],max_depth=maxd[cnt])
    clf.fit(X_train, y_train)
    Y_test_prediction = clf.predict(X_test)
    
    with open("Forest/{}_SCORE.txt".format(d), "w+") as wp:
        wp.write("Clasification report: {}".format(classification_report(y_test, Y_test_prediction)))
        wp.write('\n')
        wp.write("Confussion matrix: {}".format(confusion_matrix(y_test, Y_test_prediction)))
        wp.write('\n')
        wp.write("Train Score: {}".format(clf.score(X_train, y_train)))
        wp.write("Test Score: {}".format(clf.score(X_test, y_test)))

    
#writer.close()