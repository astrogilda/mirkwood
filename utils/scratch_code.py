
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer
from ngboost.distns import Normal
from ngboost.scores import LogScore
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
import numpy as np
from utils.custom_transformers_and_estimators import TransformerTuple, TransformerConfig, XTransformer, YTransformer, _MultipleTransformer, ReshapeTransformer, CustomNGBRegressor, CustomTransformedTargetRegressor, create_estimator
from ngboost import NGBRegressor
from utils.weightify import Weightify

from src.data_handler import DataHandler, DataHandlerConfig, DataSet, TrainData, GalaxyProperty
from src.model_handler import ModelConfig
from hypothesis.strategies import floats, lists, sampled_from, just, one_of
import pytest
from hypothesis import given, assume, settings
from pydantic import ValidationError
import numpy as np
from typing import List
import pandas as pd


def trace_calls():
    import sys
    from inspect import stack
    from copy import deepcopy

    def tracer(frame, event, arg):
        if event == "call":
            call_function = stack()[1][3]
            print(f"{call_function} called")
    sys.settrace(tracer)


# trace_calls()
create_estimator()


# Load the data
# X_y = DataHandler().get_data(train_data=TrainData.SIMBA)
# X, y = np.log10(1+X_y[0].values), X_y[1]['log_stellar_mass']
X, y = load_diabetes(return_X_y=True)


print(X.shape, y.shape)

y_transformer = YTransformer()
pipeline_y = _MultipleTransformer(y_transformer)
pipeline_y.fit(X, y)

ce = create_estimator()
ce.fit(X, y)
ce.predict_std(X).shape, y.shape

# Define your transformations
transformations_X = XTransformer(transformers=TransformerTuple([
    TransformerConfig(name="standard_scaler", transformer=StandardScaler()),
    TransformerConfig(name="robust_scaler", transformer=RobustScaler())
]))
transformations_X = XTransformer(transformation=None)


transformations_Y = YTransformer(transformers=TransformerTuple([
    TransformerConfig(name="standard_scaler", transformer=StandardScaler()),
    TransformerConfig(name="reshape_transform0",
                      transformer=ReshapeTransformer()),
]))

# Create pipelines for X and y
pipeline_X = Pipeline([(transformer.name, transformer.transformer)
                       for transformer in transformations_X.transformers])
pipeline_y = _MultipleTransformer(transformations_Y.transformers)


if transformations_X.transformers:
    feature_pipeline = Pipeline([
        ('preprocessing', pipeline_X),
        ('regressor', CustomNGBRegressor(Dist=Normal, Score=LogScore))
    ])
else:
    feature_pipeline = Pipeline([
        ('regressor', CustomNGBRegressor(Dist=Normal, Score=LogScore))
    ])


model = CustomTransformedTargetRegressor(
    regressor=feature_pipeline,
    transformer=pipeline_y
)

# Sequentially transform y
y_transformed = pipeline_y.fit_transform(y)

y_transformed.shape
y.shape
X.shape

model.fit(X, y)

y_pred_mean = model.predict(X)

y_pred_std = model.predict_std(X)


# Initialize your Weightify transformer
weightify_transformer = Weightify()

# Calculate weights for y
y_weights = weightify_transformer.fit_transform(y)

# Fit the model
model.fit(X, y, regressor__sample_weight=y_weights)

y_pred_mean_weighted = model.predict(X)

y_pred_std_weighted = model.predict_std(X)


# Sequentially transform y
y_transformed = pipeline_y.fit_transform(y)
y_transformed = pipeline_y.transform(y)

# Initialize your Weightify transformer
weightify_transformer = Weightify()

# Calculate weights for y
y_train_weights = weightify_transformer.fit_transform(y_train_transformed)
y_val_weights = weightify_transformer.transform(y_val_transformed)

# Fit the model
model.fit(X_train, y_train, regressor__sample_weight=y_train_weights, regressor__X_val=X_val,
          regressor__y_val=y_val, regressor__val_sample_weight=y_val_weights)

# Make predictions
# The predictions are automatically inverse-transformed by TransformedTargetRegressor
y_test_pred = model.predict(X_test)
