from sklearn.utils.validation import check_is_fitted
from utils.weightify import Weightify
from sklearn.pipeline import Pipeline
from utils.reshape import reshape_to_1d_array, reshape_to_2d_array
from typing import Any, List, Optional, Union, Tuple, Dict, Callable
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from copy import deepcopy
import logging
from utils.validate import validate_input, apply_transform_with_checks
from xandy_transformers import *
from customngb_regressor import ModelConfig, CustomNGBRegressor
from src.multiple_transformer import MultipleTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EPS = 1e-6


class CustomTransformedTargetRegressor(TransformedTargetRegressor):
    """
    A Custom Transformed Target Regressor
    """

    @staticmethod
    def calculate_weights(y_train: np.ndarray, y_val: Optional[np.ndarray] = None, weight_flag: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Calculate weights for training and validation.
        """
        logger.info('Calculating weights for training and validation.')
        y_train = check_array(y_train, accept_sparse=True,
                              force_all_finite='allow-nan', ensure_2d=False)
        if y_val is not None:
            y_val = check_array(y_val, accept_sparse=True,
                                force_all_finite='allow-nan', ensure_2d=False)
        weightifier = Weightify()
        if weight_flag:
            fitted_weightifier, train_weights = apply_transform_with_checks(
                transformer=weightifier, method_name='fit_transform', X=reshape_to_1d_array(y_train))
            if y_val is not None:
                val_weights = apply_transform_with_checks(
                    transformer=fitted_weightifier, method_name='transform', X=reshape_to_1d_array(y_val))
            else:
                val_weights = None
        else:
            train_weights = np.ones_like(y_train)
            val_weights = np.ones_like(y_val) if y_val is not None else None
        # train_weights = weightifier.fit_transform(y_train) if weight_flag else np.ones_like(y_train)
        # val_weights = weightifier.transform(y_val) if weight_flag and y_val is not None else None
        logger.info('Weight calculation completed.')
        return reshape_to_1d_array(train_weights), reshape_to_1d_array(val_weights) if val_weights is not None else None

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None, weight_flag: bool = False, sanity_check: bool = False) -> 'CustomTransformedTargetRegressor':
        """
        Fit the regressor.
        """
        logger.info('Fitting the CustomTransformedTargetRegressor.')
        assert X.ndim == 2, 'X must be 2d.'
        if X_val is not None:
            assert X_val.ndim == 2, 'X_val must be 2d.'

        self.transformer = apply_transform_with_checks(
            transformer=self.transformer, method_name='fit', X=reshape_to_2d_array(y), sanity_check=sanity_check)
        y = apply_transform_with_checks(
            transformer=self.transformer, method_name='transform', X=reshape_to_2d_array(y), sanity_check=sanity_check)
        y_val = apply_transform_with_checks(transformer=self.transformer, method_name='transform',
                                            X=reshape_to_2d_array(y_val), sanity_check=sanity_check) if y_val is not None else None
        # y and y_val are 2d

        train_weights, val_weights = self.calculate_weights(
            y, y_val, weight_flag)
        # train_weights and val_weights are 1d

        preprocessor = self.regressor.named_steps['preprocessor']
        preprocessor = apply_transform_with_checks(
            transformer=preprocessor, method_name='fit', X=X, sanity_check=sanity_check)
        X = apply_transform_with_checks(
            transformer=preprocessor, method_name='transform', X=X, sanity_check=sanity_check)
        X_val = apply_transform_with_checks(
            transformer=preprocessor, method_name='transform', X=X_val, sanity_check=sanity_check) if X_val is not None else None
        # X and X_val are 2d

        regressor = self.regressor.named_steps['regressor']
        regressor.fit(X=X, y=reshape_to_1d_array(y), X_val=X_val, Y_val=reshape_to_1d_array(y_val) if y_val is not None else None,
                      sample_weight=train_weights, val_sample_weight=val_weights)

        '''
        y = apply_transform_with_checks(
            transformer=self.transformer, method_name='inverse_transform', X=y, sanity_check=sanity_check)
        y_val = apply_transform_with_checks(transformer=self.transformer, method_name='inverse_transform',
                                            X=y_val, sanity_check=sanity_check) if y_val is not None else None
        y, y_val = reshape_to_1d_array(y), reshape_to_1d_array(y_val)
        X = apply_transform_with_checks(
            transformer=preprocessor, method_name='inverse_transform', X=X, sanity_check=sanity_check)
        X_val = apply_transform_with_checks(transformer=preprocessor, method_name='inverse_transform',
                                            X=X_val, sanity_check=sanity_check) if X_val is not None else None
        # y and y_val are 1d
        # X and X_val are 2d
        '''

        self.regressor_ = deepcopy(self.regressor)
        self.transformer_ = deepcopy(self.transformer)

        logger.info('CustomTransformedTargetRegressor fitted.')
        return self

    def predict(self, X: np.ndarray, sanity_check: bool = False) -> np.ndarray:
        """
        Predict the target variable.
        """
        logger.info('Predicting the target variable.')
        check_is_fitted(self, 'regressor_')
        check_is_fitted(self, 'transformer_')
        assert X.ndim == 2, 'X must be 2d.'

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
        assert X.ndim == 2, 'X must be 2d.'

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


def create_estimator(model_config: Optional[ModelConfig] = None,
                     X_transformer: Optional[XTransformer] = None,
                     y_transformer: Optional[YTransformer] = None) -> CustomTransformedTargetRegressor:
    logger.info('Creating estimator with provided configurations.')

    if model_config is not None:
        validate_input(ModelConfig, model_config=model_config)
    else:
        model_config = ModelConfig()

    if X_transformer is not None:
        validate_input(XTransformer, X_transformer=X_transformer)
    else:
        X_transformer = XTransformer()

    if y_transformer is not None:
        validate_input(YTransformer, y_transformer=y_transformer)
    else:
        y_transformer = YTransformer()

    logger.info('Building pipelines.')

    pipeline_X = Pipeline([(transformer.name, transformer.transformer)
                          for transformer in X_transformer.transformers])
    pipeline_y = MultipleTransformer(y_transformer=y_transformer)

    pipeline_steps = [('regressor', CustomNGBRegressor(**vars(model_config)))]
    if X_transformer.transformers:
        logger.info('Building feature pipeline with preprocessor.')
        pipeline_steps.insert(0, ('preprocessor', pipeline_X))

    feature_pipeline = Pipeline(pipeline_steps)

    logger.info('Building and returning CustomTransformedTargetRegressor.')

    return CustomTransformedTargetRegressor(
        regressor=feature_pipeline,
        transformer=pipeline_y
    )
