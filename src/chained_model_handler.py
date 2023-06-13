from sklearn.base import clone
import numpy as np


class NestedModel:
    def __init__(self, base_model):
        self.base_model = base_model
        self.models = []

    def fit(self, X, y_list):
        X_current = X.copy()

        for y in y_list:
            model = clone(self.base_model)
            model.fit(X_current, y)
            self.models.append(model)

            y_pred_mean = model.predict(X_current)
            y_pred_std = np.std(y - y_pred_mean)

            X_current = np.column_stack((X_current, y_pred_mean, y_pred_std))

    def predict(self, X, n=1):
        X_current = X.copy()
        y_preds = []

        for model in self.models[:n]:
            y_pred_mean = model.predict(X_current)
            y_pred_std = np.std(y_preds[-1] - y_pred_mean) if y_preds else 0

            X_current = np.column_stack((X_current, y_pred_mean, y_pred_std))
            y_preds.append(y_pred_mean)

        return y_preds[-1]

    def predict_all(self, X):
        return self.predict(X, n=len(self.models))
