import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score
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
estimators=[10,20,50,100,200]

writer= open("Forest/forestparams.txt", "w+")
scorelist=[]

for cnt, d in enumerate(datasets):
    
    X_train = pd.read_csv("../SplitData/{}_X_train.csv".format(d))
    X_test = pd.read_csv("../SplitData/{}_X_test.csv".format(d))
    y_train = pd.read_csv("../SplitData/{}_y_train.csv".format(d))
    y_train = column_or_1d(y_train, warn=True)
    y_test = pd.read_csv("../SplitData/{}_y_test.csv".format(d))
    y_test = column_or_1d(y_test, warn=True)
    max_score=0
    scorelist.clear()
    for e in estimators:
        
        clf = RandomForestClassifier(n_estimators=e, criterion=c[cnt],min_samples_split=mins[cnt],max_features=maxf[cnt],max_depth=maxd[cnt])
        y_train_proba = cross_val_predict(clf, X_train, y_train, cv=10, method="predict_proba")
        y_scores = y_train_proba[:, 1]
        score = roc_auc_score(y_train, y_scores)
        scorelist.append(score)
        if (score > max_score):
            bestsParams[d]={
                    "n_estimators":e,
                    "ROC":score
                    }
            max_score = score
    
    with open("Forest/{}.txt".format(d), "w+") as wp:
        for x in scorelist:
            wp.write(str(x))
            wp.write('\n')
            
    print(d)
    print('\n')
    print(str(bestsParams[d]))
    print('\n')

for i in bestsParams:
    writer.write(str(i))
    writer.write('\n')
    writer.write(str(bestsParams[i]))
    writer.write('\n')
    
writer.close()