import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.ensemble import BaggingClassifier
from sklearn.utils.validation import column_or_1d
import matplotlib.pyplot as plt
import seaborn as sns

name = "a_wh_question"

X_train = pd.read_csv("../SplitPreprocessedData/" +
                      name + "_X_train.csv")
X_test = pd.read_csv("../SplitPreprocessedData/" +
                     name + "_X_test.csv")
y_train = pd.read_csv("../SplitPreprocessedData/" +
                      name + "_y_train.csv")
y_train = column_or_1d(y_train, warn=True)
y_test = pd.read_csv("../SplitPreprocessedData/" +
                     name + "_y_test.csv")
y_test = column_or_1d(y_test, warn=True)

mlp = MLPClassifier(random_state=42, hidden_layer_sizes=(30, 30), max_iter=50,
                    activation="relu", solver="adam", alpha=0.0003, learning_rate_init=0.01)

bag = BaggingClassifier(mlp, n_estimators=10)

bag.fit(X_train, y_train)
y_pred = bag.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, center=True)
plt.show()
print(classification_report(y_test, y_pred))
