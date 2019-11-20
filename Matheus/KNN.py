import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


datasets = ["a_affirmative", "a_conditional", "a_doubt_question", "a_emphasis", "a_negative", "a_relative", "a_topics", "a_wh_question", "a_yn_question",
         "b_affirmative", "b_conditional", "b_doubt_question", "b_emphasis", "b_negative", "b_relative", "b_topics", "b_wh_question", "b_yn_question"]


for dataset in datasets:
    df = pd.read_csv("../PreprocessedDataset/" + dataset + ".csv", sep=" ")
    bestRocAucScore = 0
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    k_range = range(1, 15)
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)        
        scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
        k_scores.append(scores.mean())
    plt.plot(k_range, k_scores, label=dataset)
    plt.suptitle(dataset)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()

