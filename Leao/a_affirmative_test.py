import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.utils.validation import column_or_1d

X = pd.read_csv("../dataset/" + "a_affirmative" + "_datapoints.csv", sep=' ')
y = pd.read_csv("../dataset/" + "a_affirmative" + "_targets.csv")
y = column_or_1d(y, warn=True)
X.drop([X.columns[0]], axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.3)

mlp = MLPClassifier(random_state=42, hidden_layer_sizes=(250, 250, 250), max_iter=50,
                    activation="relu", solver="adam", alpha=0.0001, learning_rate_init=0.001)

y_train_proba_mlp = cross_val_predict(
    mlp, X_train, y_train, cv=10, method="predict_proba")
y_scores_mlp = y_train_proba_mlp[:, 1]
MLPScore = roc_auc_score(y_train, y_scores_mlp)


print(MLPScore)