import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import pandas as pd
from sklearn import tree
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

clf1 = KNeighborsClassifier()
clf2 = RandomForestClassifier()
clf3 = tree.DecisionTreeClassifier()


datasets = ["a_affirmative.csv", "a_conditional.csv", "a_doubt_question.csv", "a_emphasis.csv", "a_negative.csv", "a_relative.csv", "a_topics.csv", "a_wh_question.csv", "a_yn_question.csv",
             "b_affirmative.csv", "b_conditional.csv", "b_doubt_question.csv", "b_emphasis.csv", "b_negative.csv", "b_relative.csv", "b_topics.csv", "b_wh_question.csv"]
labels = ['KNeighbors', 'Random Forest', 'Decision Tree']


for d in datasets:
    print(d)
    df = pd.read_csv("../PreprocessedDataset/a_affirmative.csv", sep=" ")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    #for clf, label in zip([clf1, clf2, clf3], labels):
        #scores = model_selection.cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
        #print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
        

    eclf1 = VotingClassifier(estimators=[
            ('knn', clf1), ('rf', clf2), ('td', clf3)], voting='hard')
    eclf1 = eclf1.fit(X_train, y_train)
    y_test_pred = eclf1.predict(X_test)
    print(log_loss(y_test, y_test_pred))


    eclf2 = VotingClassifier(estimators=[
            ('knn', clf1), ('rf', clf2), ('td', clf3)],
            voting='soft')
    eclf2 = eclf2.fit(X_train, y_train)
    y_test_pred = eclf2.predict_proba(X_test)
    print(log_loss(y_test, y_test_pred))
