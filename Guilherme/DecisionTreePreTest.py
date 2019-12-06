import pandas as pd
from time import perf_counter
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

maxd= [20,10,30,30,10,50,10,None,20,20,20,None,None,20,50,20,30,30]
maxf= [4,18,'log2','log2','auto','auto','auto','auto','auto','log2','sqrt',8,4,'log2',None,'log2',14,8]
mins= [12,12,20,20,12,20,20,12,20,20,20,20,12,20,20,20,20,20]
c= ["gini", "entropy","entropy","entropy","gini","gini","gini","entropy","gini","entropy","gini","entropy","gini","entropy","entropy","entropy","gini","gini"]
e=[200,100,200,200,200,200,200,100,200,200,200,200,100,100,200,200,100,100]

#writer= open("Forest/forestSCORE.txt", "w+")
#scorelist=[]

for cnt, d in enumerate(datasets):
    
    X_train = pd.read_csv("../SplitPreprocessedData/{}_X_train.csv".format(d))
    X_test = pd.read_csv("../SplitPreprocessedData/{}_X_test.csv".format(d))
    y_train = pd.read_csv("../SplitPreprocessedData/{}_y_train.csv".format(d))
    y_train = column_or_1d(y_train, warn=False)
    y_test = pd.read_csv("../SplitPreprocessedData/{}_y_test.csv".format(d))
    y_test = column_or_1d(y_test, warn=False)
    seconds = perf_counter()
    clf = tree.DecisionTreeClassifier(criterion=c[cnt],min_samples_split=mins[cnt],max_features=maxf[cnt],max_depth=maxd[cnt])
    clf.fit(X_train, y_train)
    Y_test_prediction = clf.predict(X_test)
    print(perf_counter() - seconds)
    
    # with open("DecisionTreePRE2/{}_SCORE.txt".format(d), "w+") as wp:
    #     wp.write("Clasification report: {}".format(classification_report(y_test, Y_test_prediction)))
    #     wp.write('\n')
    #     wp.write("Confussion matrix: {}".format(confusion_matrix(y_test, Y_test_prediction)))
    #     wp.write('\n')
    #     wp.write("Train Score: {}".format(clf.score(X_train, y_train)))
    #     wp.write("Test Score: {}".format(clf.score(X_test, y_test)))

    
#writer.close()