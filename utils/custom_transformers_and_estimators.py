from typing import List, Optional, Dict
import numpy as np
from sklearn.base import TransformerMixin
from ngboost import NGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import TransformedTargetRegressor


class MultipleTransformer(TransformerMixin):
    """
    A transformer that applies a list of transformers sequentially.

    Parameters
    ----------
    transformers : List[TransformerMixin]
        A list of TransformerMixin objects to be applied sequentially.
    """

    def __init__(self, transformers: List[TransformerMixin]):
        self.transformers = transformers

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'MultipleTransformer':
        """Fit all transformers using X and y."""
        for transformer in self.transformers:
            transformer.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X using all transformers."""
        result = X
        for transformer in self.transformers:
            result = transformer.transform(result)
        return result

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform X using all transformers."""
        result = X
        for transformer in reversed(self.transformers):
            result = transformer.inverse_transform(result)
        return result


class CustomNGBRegressor(NGBRegressor):
    """
    A custom NGBRegressor class compatible with scikit-learn Pipeline.
    Inherits from NGBRegressor and overrides the fit and predict methods to work with Pipeline.
    """

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None, sample_weight: Optional[np.ndarray] = None,
            val_sample_weight: Optional[np.ndarray] = None) -> 'CustomNGBRegressor':
        """Fit the model according to the given training data."""
        return super().fit(X, y, X_val=X_val, y_val=y_val,
                           sample_weight=sample_weight, val_sample_weight=val_sample_weight)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the base model and return the mean of the predicted distribution."""
        dist = super().pred_dist(X)
        return dist.loc

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """Predict using the base model and return the standard deviation of the predicted distribution."""
        dist = super().pred_dist(X)
        return dist.scale

    @property
    def base_model(self):
        return self


'''
# Define your transformations
transformations_X = [("standard_scaler", StandardScaler())]
transformations_y = [
    ("log_transform", FunctionTransformer(np.log1p, inverse_func=np.expm1)),
    ("sqrt_transform", FunctionScaler(np.sqrt, inverse_func=np.square)),
]

# Create pipelines for X and y
pipeline_X = Pipeline(transformations_X)
pipeline_y = MultipleTransformer(transformations_y)

feature_pipeline = Pipeline([
    ('preprocessing', pipeline_X),
    ('regressor', CustomNGBRegressor(Dist=RegressionDistn, Score=LogScore))
])

model = TransformedTargetRegressor(
    regressor=feature_pipeline,
    transformer=pipeline_y
)

# Sequentially transform y
y_train_transformed = pipeline_y.fit_transform(y_train)
y_val_transformed = pipeline_y.transform(y_val)

# Initialize your Weightify transformer
weightify_transformer = Weightify()

# Calculate weights for y
y_train_weights = weightify_transformer.fit_transform(y_train_transformed)
y_val_weights = weightify_transformer.transform(y_val_transformed)

# Fit the model
model.fit(X_train, y_train_transformed, regressor__sample_weight=y_train_weights, regressor__X_val=X_val,
          regressor__y_val=y_val_transformed, regressor__val_sample_weight=y_val_weights)

# Make predictions
# The predictions are automatically inverse-transformed by TransformedTargetRegressor
y_test_pred = model.predict(X_test)
'''
