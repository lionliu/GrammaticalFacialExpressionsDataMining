import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import column_or_1d
import matplotlib.pyplot as plt
import seaborn as sns

name = "b_relative"

X_train = pd.read_csv("../SplitPreprocessedData/" +
                      name + "_X_train.csv")
X_test = pd.read_csv("../SplitPreprocessedData/" +
                     name + "_X_test.csv")
y_train = pd.read_csv("../SplitPreprocessedData/" +
                      name + "_y_train.csv")
y_train = column_or_1d(y_train, warn=False)
y_test = pd.read_csv("../SplitPreprocessedData/" +
                     name + "_y_test.csv")
y_test = column_or_1d(y_test, warn=False)

knn = KNeighborsClassifier(n_neighbors=4, weights='distance',
                           algorithm='ball_tree', leaf_size=1)


knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, center=True)
plt.show()
print(classification_report(y_test, y_pred))



