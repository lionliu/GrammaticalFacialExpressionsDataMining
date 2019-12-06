import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.validation import column_or_1d
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

datasets = ["a_affirmative", "a_conditional", "a_doubt_question", "a_emphasis", "a_negative", "a_relative", "a_topics", "a_wh_question", "a_yn_question",
             "b_affirmative", "b_conditional", "b_doubt_question", "b_emphasis", "b_negative", "b_relative", "b_topics", "b_wh_question", "b_yn_question"]

bestsParams = {}

maxd= [20,10,30,30,10,50,10,None,20,20,20,None,None,20,50,20,30,30]
maxf= [4,18,'log2','log2','auto','auto','auto','auto','auto','log2','sqrt',8,4,'log2',None,'log2',14,8]
mins= [12,12,20,20,12,20,20,12,20,20,20,20,12,20,20,20,20,20]
c= ["gini", "entropy","entropy","entropy","gini","gini","gini","entropy","gini","entropy","gini","entropy","gini","entropy","entropy","entropy","gini","gini"]
estimators=[10,20,50,100,200]

# writer= open("ForestPREPROCESSADO/forestparams.txt", "w+")


scorelist = []
combinations = range(0, 5)  # N de combinacoes

plt.title("ROC AUC Score Combinations for Random Forest")
for cnt, d in enumerate(datasets):
    
    X_train = pd.read_csv("../SplitPreprocessedData/{}_X_train.csv".format(d))
    X_test = pd.read_csv("../SplitPreprocessedData/{}_X_test.csv".format(d))
    y_train = pd.read_csv("../SplitPreprocessedData/{}_y_train.csv".format(d))
    y_train = column_or_1d(y_train, warn=True)
    y_test = pd.read_csv("../SplitPreprocessedData/{}_y_test.csv".format(d))
    y_test = column_or_1d(y_test, warn=True)
    # max_score=0
    # scorelist.clear()
    roc = []
    for e in estimators:
        
        clf = RandomForestClassifier(n_estimators=e, criterion=c[cnt],min_samples_split=mins[cnt],max_features=maxf[cnt],max_depth=maxd[cnt])
        y_train_proba = cross_val_predict(clf, X_train, y_train, cv=10, method="predict_proba")
        y_scores = y_train_proba[:, 1]
        score = roc_auc_score(y_train, y_scores)
        roc.append(score)
        # if (score > max_score):
        #     bestsParams[d]={
        #             "n_estimators":e,
        #             "ROC":score
        #             }
        #     max_score = score
    plt.plot(combinations, roc, label=d)
    # with open("ForestPREPROCESSADO/{}.txt".format(d), "w+") as wp:
    #     for x in scorelist:
    #         wp.write(str(x))
    #         wp.write('\n')
            
    # print(d)
    # print('\n')
    # print(str(bestsParams[d]))
    # print('\n')

# for i in bestsParams:
#     writer.write(str(i))
#     writer.write('\n')
#     writer.write(str(bestsParams[i]))
#     writer.write('\n')
    
# writer.close()

plt.legend()
plt.ylabel("ROC AUC")
plt.xlabel("Combination")
plt.xticks(list(combinations))
plt.legend()

plt.show()
