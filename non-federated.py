from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(random_state=0, solver="liblinear")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(f"Accuracy = {accuracy_score(y_test, y_pred)}")