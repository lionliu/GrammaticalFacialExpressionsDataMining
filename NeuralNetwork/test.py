import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

df = pd.read_csv("../PreprocessedDataset/" + "b_wh_question" + ".csv", sep=" ")
bestRocAucScore = 0
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.3)

mlp = MLPClassifier(random_state=42, hidden_layer_sizes=(
    250, 250, 250), max_iter=50, activation="relu", solver="adam", alpha=0.003, learning_rate_init=0.003)
y_train_proba = cross_val_predict(
    mlp, X_train, y_train, cv=10, method="predict_proba")
y_scores = y_train_proba[:, 1]
fpr, tpr, threshold = roc_curve(y_train, y_scores)
# print(y_train_proba)
# print(roc_auc_score(y_train, y_scores))
# y_scores = y_train_proba[:, 1]


def plot_roc_curve(fpr, tpr, label=None): 
    plt.plot(fpr, tpr, linewidth=2, label=label) 
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16)  # Not shown
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    # Not shown
    plt.grid(True)                                            # Not shown
plot_roc_curve(fpr, tpr)
plt.show()
