import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.validation import column_or_1d
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn import tree

datasets = ["a_affirmative", "a_conditional", "a_doubt_question", "a_emphasis", "a_negative", "a_relative", "a_topics", "a_wh_question", "a_yn_question",
             "b_affirmative", "b_conditional", "b_doubt_question", "b_emphasis", "b_negative", "b_relative", "b_topics", "b_wh_question"]
d="a_affirmative"

bestsParams = {}

featuren=[]
max_depth= [1,2,3,4,5,6,7,8,9,10, None]
max_features= [1,2,3,4,5,6,7,8,9,10,11,12, 'auto', 'sqrt', 'log2', None]
min_samples_split= [2, 3,4,5,6,7,8,9, 10]
criterion= ["gini", "entropy"]

#writer= open("AAA.txt", "w+")

X_train = pd.read_csv("../SplitData/{}_X_train.csv".format(d))
X_test = pd.read_csv("../SplitData/{}_X_test.csv".format(d))
y_train = pd.read_csv("../SplitData/{}_y_train.csv".format(d))
y_train = column_or_1d(y_train, warn=True)
y_test = pd.read_csv("../SplitData/{}_y_test.csv".format(d))
y_test = column_or_1d(y_test, warn=True)

for col in X_train.columns:
    featuren.append(col)
    
print(str(featuren))

clf = tree.DecisionTreeClassifier(criterion='entropy',min_samples_split=7,max_features=10,max_depth=6)
clf.fit(X_train, y_train)
preplot = open("plot.dot", 'w+')
tree.export_graphviz(clf, out_file=preplot, feature_names=featuren)
preplot.close()