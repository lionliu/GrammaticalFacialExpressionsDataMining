import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.validation import column_or_1d
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn import tree

datasets = ["a_affirmative", "a_conditional", "a_doubt_question", "a_emphasis", "a_negative", "a_relative", "a_topics", "a_wh_question", "a_yn_question",
             "b_affirmative", "b_conditional", "b_doubt_question", "b_emphasis", "b_negative", "b_relative", "b_topics", "b_wh_question", "b_yn_question"]

bestsParams = {}

max_depth= [5,10,20,30,50,100, None]
max_features= [4,8,14,20,40,60,80, 'auto', 'sqrt', 'log2', None]
min_samples_split= [2,6,12,20]
criterion= ["gini", "entropy"]

writer= open("DecisionTree2/decisiontreeparam.txt", "w+")

#with open("DecisionTreePRE2/abc.txt".format(d), "w+") as wp:
#    for x in scorelist:
#        wp.write(str(x))
#        wp.write('\n')

scorelist=[]

for d in datasets:
    
    X_train = pd.read_csv("../SplitData/{}_X_train.csv".format(d))
    X_test = pd.read_csv("../SplitData/{}_X_test.csv".format(d))
    y_train = pd.read_csv("../SplitData/{}_y_train.csv".format(d))
    y_train = column_or_1d(y_train, warn=True)
    y_test = pd.read_csv("../SplitData/{}_y_test.csv".format(d))
    y_test = column_or_1d(y_test, warn=True)
    max_score=0
    for c in criterion:
        for mins in min_samples_split:
            for maxf in max_features:
                for maxd in max_depth:
                    clf = tree.DecisionTreeClassifier(criterion=c,min_samples_split=mins,max_features=maxf,max_depth=maxd)
                    #print(str(maxf))
                    y_train_proba = cross_val_predict(clf, X_train, y_train, cv=10, method="predict_proba")
                    y_scores = y_train_proba[:, 1]
                    score = roc_auc_score(y_train, y_scores)
                    scorelist.append(score)
                    if (score > max_score):
                        bestsParams[d]={
                                "criterion":c,
                                "max_depth":maxd,
                                "max_features":maxf,
                                "min_samples_split":mins,
                                "ROC":score
                                }
                        max_score = score
                        
    with open("DecisionTree2/{}.txt".format(d), "w+") as wp:
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