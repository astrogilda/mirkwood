from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.base import TransformerMixin
import logging
import numpy as np
from copy import deepcopy
from typing import Any, List, Optional, Tuple, Dict, Callable, Union
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y
from sklearn.compose import TransformedTargetRegressor
from utils.weightify import Weightify
from utils.validate import validate_input
from utils.transform_with_checks import apply_transform_with_checks
from utils.reshape import reshape_to_1d_array, reshape_to_2d_array
from transformers.xandy_transformers import XTransformer, YTransformer
from regressors.customngb_regressor import ModelConfig, CustomNGBRegressor
from transformers.multiple_transformer import MultipleTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EPS = 1e-6


class CustomTransformedTargetRegressor(TransformedTargetRegressor):
    """
    A Custom Transformed Target Regressor
    """

    def __init__(self,
                 regressor: BaseEstimator = None,
                 transformer: TransformerMixin = None,
                 check_inverse: bool = True,
                 weightifier: Weightify = None):
        super().__init__(regressor=regressor,
                         transformer=transformer,
                         check_inverse=check_inverse)
        self.regressor = regressor
        self.transformer = transformer
        self.weightifier = weightifier or Weightify()

    def _calculate_weights(self, y_train: np.ndarray, y_val: Optional[np.ndarray] = None) -> np.ndarray:
        if self.weightifier is None:
            return None
        else:
            y = y_train if y_val is None else np.concatenate([y_train, y_val])
            weights = apply_transform_with_checks(
                transformer=self.weightifier, method_name='transform', X=reshape_to_2d_array(y))
            return weights

    def _calculate_weights(self, y_train: np.ndarray, y_val: Optional[np.ndarray] = None, weight_flag: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Calculate weights for training and validation.
        """
        if weight_flag:
            logger.info('Calculating weights for training and validation.')

            self.weightifier, train_weights = apply_transform_with_checks(
                transformer=self.weightifier, method_name='fit_transform', X=reshape_to_1d_array(y_train))
            if y_val is not None:
                val_weights = apply_transform_with_checks(
                    transformer=self.weightifier, method_name='transform', X=reshape_to_1d_array(y_val))
            else:
                val_weights = None
            logger.info('Weight calculation completed.')

        else:
            train_weights = np.ones_like(y_train)
            val_weights = np.ones_like(y_val) if y_val is not None else None
        return reshape_to_1d_array(train_weights), reshape_to_1d_array(val_weights) if val_weights is not None else None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None, **fit_params) -> 'CustomTransformedTargetRegressor':
        y_train_trans = apply_transform_with_checks(
            transformer=self.transformer, method_name='transform', X=reshape_to_2d_array(y_train))
        weights = self._calculate_weights(y_train, y_val)

        self.regressor_ = clone(self.regressor)

        if isinstance(self.regressor_, Pipeline):
            self.regressor_.fit(X_train, y_train_trans,
                                regressor__sample_weight=weights)
        else:
            self.regressor_.fit(X_train, y_train_trans, sample_weight=weights)

        self.transformer_ = self.transformer
        return self

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> 'CustomTransformedTargetRegressor':
        """
        Fit the regressor.
        """
        logger.info('Fitting the CustomTransformedTargetRegressor.')

        X, y = check_X_y(X, y, accept_sparse=True,
                         force_all_finite=True, y_numeric=True)

        X_val = fit_params.pop("X_val", None)
        y_val = fit_params.pop("y_val", None)

        if X_val is not None and y_val is not None:
            X_val, y_val = check_X_y(
                X_val, y_val, accept_sparse=True, force_all_finite=True, y_numeric=True)

        weight_flag = fit_params.pop("weight_flag", False)
        sanity_check = fit_params.pop("sanity_check", False)

        self.transformer, y = apply_transform_with_checks(
            transformer=self.transformer, method_name='fit_transform', X=reshape_to_2d_array(y), sanity_check=sanity_check)
        y_val = apply_transform_with_checks(transformer=self.transformer, method_name='transform',
                                            X=reshape_to_2d_array(y_val), sanity_check=sanity_check) if y_val is not None else None
        # y and y_val are 2d

        train_weights, val_weights = self._calculate_weights(
            y, y_val, weight_flag)
        # train_weights and val_weights are 1d

        if "X_val" in fit_params:
            fit_params.pop("X_val")
        if "y_val" in fit_params:
            fit_params.pop("y_val")

        if "sample_weight" not in fit_params:
            fit_params["sample_weight"] = train_weights
        if "val_sample_weight" not in fit_params:
            fit_params["val_sample_weight"] = val_weights

        self.transformer_ = clone(self.transformer)

        self.regressor_ = clone(self.regressor)

        if isinstance(self.regressor_, Pipeline):
            if 'preprocessor' in self.regressor_.named_steps:
                self.regressor_.named_steps['preprocessor'] = apply_transform_with_checks(
                    transformer=self.regressor_.named_steps['preprocessor'], method_name='fit', X=X, sanity_check=sanity_check)

            self.regressor_.named_steps['regressor'] = apply_transform_with_checks(
                transformer=self.regressor_.named_steps['regressor'], method_name='fit', X=X, y=y, X_val=X_val, y_val=y_val, sanity_check=sanity_check)

        else:
            self.regressor_ = apply_transform_with_checks(
                transformer=self.regressor_, method_name='fit', X=X, y=y, X_val=X_val, y_val=y_val, sanity_check=sanity_check)

        logger.info('Fitting completed.')
        return self

    def predict(self, X: np.ndarray, sanity_check: bool = False) -> np.ndarray:
        """
        Predict the target variable.
        """
        logger.info('Predicting the target variable.')
        check_is_fitted(self, 'regressor_')
        check_is_fitted(self, 'transformer_')
        X = check_array(X, accept_sparse=True,
                        force_all_finite=True, ensure_2d=True)

        X_trans = apply_transform_with_checks(
            transformer=self.regressor_.named_steps['preprocessor'], method_name='transform', X=X, sanity_check=sanity_check)
        # X_trans is 2d
        y_pred_mean = apply_transform_with_checks(
            transformer=self.regressor_.named_steps['regressor'], method_name='predict', X=X_trans, sanity_check=sanity_check)
        # y_pred_mean can be 1d or 2d, I don't know
        y_pred_mean = apply_transform_with_checks(
            transformer=self.transformer_, method_name='inverse_transform', X=reshape_to_2d_array(y_pred_mean), sanity_check=sanity_check)
        # y_pred_mean is 2d
        logger.info('Prediction completed.')
        return reshape_to_1d_array(y_pred_mean)

    def predict_std(self, X: np.ndarray, sanity_check: bool = False) -> np.ndarray:
        """
        Predict the standard deviation of the target variable.
        """
        logger.info('Predicting the standard deviation of the target variable.')
        check_is_fitted(self, 'regressor_')
        check_is_fitted(self, 'transformer_')
        X = check_array(X, accept_sparse=True,
                        force_all_finite=True, ensure_2d=True)

        X_trans = apply_transform_with_checks(
            transformer=self.regressor_.named_steps['preprocessor'], method_name='transform', X=X, sanity_check=sanity_check)
        # X_trans is 2d
        y_pred_mean = apply_transform_with_checks(
            transformer=self.regressor_.named_steps['regressor'], method_name='predict', X=X_trans, sanity_check=sanity_check)
        # y_pred_mean can be 1d or 2d, I don't know
        y_pred_std = apply_transform_with_checks(
            transformer=self.regressor_.named_steps['regressor'], method_name='predict_std', X=X_trans, sanity_check=sanity_check)
        # y_pred_std can be 1d or 2d, I don't know

        y_pred_upper = y_pred_mean + y_pred_std
        y_pred_lower = y_pred_mean - y_pred_std
        # y_pred_upper and y_pred_lower can be 1d or 2d, I don't know

        y_pred_upper_inverse = apply_transform_with_checks(
            transformer=self.transformer_, method_name='inverse_transform', X=reshape_to_2d_array(y_pred_upper), sanity_check=sanity_check)
        y_pred_lower_inverse = apply_transform_with_checks(
            transformer=self.transformer_, method_name='inverse_transform', X=reshape_to_2d_array(y_pred_lower), sanity_check=sanity_check)
        y_pred_std_inverse = (y_pred_upper_inverse - y_pred_lower_inverse) / 2
        # y_pred_std_inverse is 2d

        logger.info('Prediction of standard deviation completed.')
        return reshape_to_1d_array(y_pred_std_inverse)

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> float:
        """
        Score using the fitted regressor.
        """
        logger.info('Scoring using the CustomTransformedTargetRegressor.')
        y_pred = self.predict(X)

        if isinstance(self.regressor_, Pipeline) and 'regressor' in self.regressor_.named_steps:
            score = self.regressor_.named_steps['regressor'].score(
                X, y_pred, sample_weight=sample_weight)
        else:
            score = self.regressor_.score(
                X, y_pred, sample_weight=sample_weight)

        return score

    def get_params(self, deep=True):
        return {
            **super().get_params(deep),
            "weightifier": self.weightifier
        }

    def set_params(self, **parameters):
        if 'weightifier' in parameters:
            setattr(self, 'weightifier', parameters.pop('weightifier'))
        super().set_params(**parameters)


def create_estimator(model_config: Optional[ModelConfig] = None,
                     X_transformer: Optional[XTransformer] = None,
                     y_transformer: Optional[YTransformer] = None, weightifier: Optional[Weightify] = None) -> CustomTransformedTargetRegressor:
    """
    Create an estimator using custom transformations.
    """
    logger.info('Creating estimator with provided configurations.')

    if model_config is not None:
        pass
        # validate_input(ModelConfig, model_config=model_config)
    else:
        model_config = ModelConfig()

    if X_transformer is not None:
        validate_input(XTransformer, arg=X_transformer)
    else:
        X_transformer = XTransformer()

    if y_transformer is not None:
        validate_input(YTransformer, y_transformer=y_transformer)
    else:
        y_transformer = YTransformer()

    if weightifier is not None:
        validate_input(Weightify, weightifier=weightifier)
    else:
        weightifier = Weightify()

    logger.info('Building pipelines.')

    pipeline_X = Pipeline([(transformer.name, transformer.transformer)
                          for transformer in X_transformer.transformers])
    pipeline_y = MultipleTransformer(**vars(y_transformer))

    pipeline_steps = [('regressor', CustomNGBRegressor(**vars(model_config)))]
    if X_transformer.transformers:
        logger.info('Building feature pipeline with preprocessor.')
        pipeline_steps.insert(0, ('preprocessor', pipeline_X))

    feature_pipeline = Pipeline(pipeline_steps)

    ctt_regressor = CustomTransformedTargetRegressor(
        regressor=feature_pipeline,
        transformer=pipeline_y
    )

    logger.info('CustomTransformedTargetRegressor created.')
    return ctt_regressor
