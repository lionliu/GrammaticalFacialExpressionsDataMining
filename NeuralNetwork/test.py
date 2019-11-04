import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

df = pd.read_csv("../PreprocessedDataset/" + "a_affirmative" + ".csv", sep=" ")
bestRocAucScore = 0
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.3)

mlp = MLPClassifier()
rf = RandomForestClassifier()
y_train_proba = cross_val_predict(
    mlp, X_train, y_train, cv=10, method="predict_proba")
y_pred = cross_val_predict(
    mlp, X_train, y_train, cv=10, )
print(y_train_proba)
y_scores = y_train_proba[:, 1]
print(y_scores)
print(roc_auc_score(y_train, y_scores))
print(y_pred)
print(roc_auc_score(y_train, y_pred))
# y_scores = y_train_proba[:, 1]

