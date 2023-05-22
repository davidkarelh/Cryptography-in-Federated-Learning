from numpy import ndarray
from sklearn.model_selection import train_test_split
from Node import Node
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class Client(Node):
    _server: Node
    _model: LogisticRegression
    _X_train: ndarray
    _y_train: ndarray
    _X_test: ndarray
    _y_test: ndarray

    def __init__(self) -> None:
        super().__init__()
    
    def set_server(self, server):
        self._server = server
    
    def initialize_model(self, random_state, solver):
        self._model = LogisticRegression(random_state=random_state, solver=solver)

    def set_coef_and_intercept(self, coef: ndarray, intercept: ndarray):
        if hasattr(self._model, "coef_"):
            self._model.coef_ = coef
            self._model.intercept_ = intercept
    
    def set_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test
    
    def train_and_return_weights(self):
        old_coefs:ndarray
        old_intercepts: ndarray
        old_coefs_exist = True

        if hasattr(self._model, "coef_"):
            old_coefs = self._model.coef_.copy()
            old_intercepts = self._model.intercept_.copy()
        else:
            old_coefs_exist = False

        self._model.fit(self._X_train, self._y_train)
        data_len = len(self._y_train)

        y_pred = self._model.predict(self._X_test)
        print(f"Local Accuracy = {accuracy_score(self._y_test, y_pred)}")

        if old_coefs_exist:
            secret_values = [data_len, data_len * (self._model.intercept_[0] - old_intercepts[0])]
            for element in (data_len * (self._model.coef_ - old_coefs))[0]:
                secret_values.append(element)

            self.set_secret_values(secret_values)
            return data_len * (self._model.coef_ - old_coefs), data_len, data_len * (self._model.intercept_ - old_intercepts)
        else:
            secret_values = [data_len, data_len * (self._model.intercept_[0])]
            for element in (data_len * (self._model.coef_))[0]:
                secret_values.append(element)

            self.set_secret_values(secret_values)
            return data_len * (self._model.coef_), data_len, data_len * (self._model.intercept_)

    def evaluate_on_local_data(self):
        y_pred = self._model.predict(self._X_test)
        print(f"Local Accuracy = {accuracy_score(self._y_test, y_pred)}")