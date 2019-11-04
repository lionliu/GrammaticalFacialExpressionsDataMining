import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("../PreprocessedDataset/b_wh_question.csv", sep=" ")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

mlp = MLPClassifier(random_state=42, hidden_layer_sizes=(250, 250, 250), max_iter=50, solver="adam",
                    learning_rate_init=0.003,
                    alpha=0.0003,
                    activation="logistic")


mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

r = classification_report(y_test, y_pred)
print(r)
