
from src.data_handler import GalaxyProperty
from utils.weightify import Weightify
from sklearn.pipeline import Pipeline
from utils.reshape import reshape_to_1d_array, reshape_to_2d_array
from typing import Any, List, Optional, Union, Tuple, Dict, Callable
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from pydantic import BaseModel, Field, validator, root_validator, conint, confloat
from ngboost.scores import LogScore, Score
from ngboost.distns import Normal, Distn
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.base import BaseEstimator, TransformerMixin, clone
from ngboost import NGBRegressor
import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn.compose import TransformedTargetRegressor
from copy import deepcopy
import logging
from utils.validate import validate_input

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EPS = 1e-6

# TODO: add sanity checks for transformers other than StandardScaler

# Centralizing the model parameters in a model config class for better readability and maintainability


class ModelConfig(BaseModel):
    """
    Model configuration for the NGBRegressor. We use this to have a
    centralized place for model parameters which enhances readability
    and maintainability.
    """
    Base: DecisionTreeRegressor = Field(
        default=DecisionTreeRegressor(
            criterion='friedman_mse',
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_leaf_nodes=31,
            splitter='best'),
        description="Base learner for NGBRegressor"
    )
    Dist: Distn = Normal
    Score: Score = LogScore
    n_estimators: conint(gt=0) = Field(
        default=500, description="Number of estimators for NGBRegressor")
    learning_rate: confloat(gt=0, le=1) = Field(
        default=0.04,
        description="The learning rate for the NGBRegressor. Must be greater than 0 and less than or equal to 1."
    )
    col_sample: confloat(gt=0, le=1) = Field(
        default=1.0,
        description="The column sample rate. Must be greater than 0 and less than or equal to 1."
    )
    minibatch_frac: confloat(gt=0, le=1) = Field(
        default=1.0,
        description="The minibatch fraction for NGBRegressor. Must be greater than 0 and less than or equal to 1."
    )
    verbose: bool = False
    natural_gradient: bool = True
    early_stopping_rounds: Optional[conint(gt=0)] = Field(
        default=None, description="Early stopping rounds for NGBRegressor")

    class Config:
        arbitrary_types_allowed: bool = True


class TransformerConfig(BaseModel):
    """
    Config for a transformer.
    """
    name: str
    transformer: TransformerMixin

    class Config:
        arbitrary_types_allowed: bool = True


class TransformerTuple(list):
    """
    This class represents a list of transformers.
    """

    def __init__(self, transformers):
        if isinstance(transformers, TransformerConfig):
            transformers = [transformers]

        super().__init__(transformers)


class TransformerBase(BaseModel):
    """
    Base class for Transformers handling common validations.
    """
    transformers: Optional[Union[TransformerTuple, TransformerConfig]] = Field(
        default=None
    )

    @root_validator(pre=True)
    def validate_transformers(cls, values):
        logger.info('Validating transformers in TransformerBase.')
        transformers = values.get("transformers")

        # If transformers is an instance of a TransformerMixin but not a TransformerConfig, raise an error
        if isinstance(transformers, TransformerMixin) and not isinstance(transformers, TransformerConfig):
            raise ValueError(
                "transformers must be of type TransformerConfig or TransformerTuple")

        if transformers is None:
            transformers = TransformerTuple([
                TransformerConfig(name="standard_scaler",
                                  transformer=StandardScaler())
            ])
        elif isinstance(transformers, TransformerConfig):
            transformers = [transformers]

        values["transformers"] = transformers
        logger.info('Transformers validated.')
        return values


class XTransformer(TransformerBase):
    """
    This class handles transformers for X.
    """
    pass


class YTransformer(TransformerBase):
    """
    This class handles transformers for y.
    """
    pass


def apply_transform_with_checks(transformer: TransformerMixin, method_name: str, X: np.ndarray, y: Optional[np.ndarray] = None, sanity_check: bool = False, **kwargs) -> Union[np.ndarray, TransformerMixin]:
    logger.info(
        f'Applying transformation using {transformer.__class__.__name__}.')

    valid_methods = ['transform', 'fit', 'fit_transform',
                     'inverse_transform', 'predict', 'predict_std']
    if method_name not in valid_methods:
        raise ValueError(
            f"Invalid method name: {method_name}. Must be one of {valid_methods}")

    method = getattr(transformer, method_name)

    try:
        if y is None:
            transformed_data = method(X)
        else:
            transformed_data = method(X, y)
    except Exception as e:
        raise ValueError(
            f"Failed to transform data with {transformer.__class__.__name__}. Original error: {e}")

    if sanity_check:
        if method_name in ['transform', 'fit_transform']:
            inverse_transformed_data = transformer.inverse_transform(
                transformed_data)
            assert np.allclose(
                a=X, b=inverse_transformed_data, rtol=0.05, atol=1e-10), 'The inverse transformation correctly reverts the data.'
        elif method_name == 'inverse_transform' and transformer.__class__.__name__ == 'StandardScaler':
            transformed_mean = np.mean(transformed_data, axis=0)
            transformer_mean = transformer.mean_
            mean_check = np.allclose(
                transformed_mean, transformer_mean, rtol=EPS, atol=EPS)
            if not np.allclose(X, np.mean(X, axis=0), rtol=1e-1):
                transformed_std = np.std(transformed_data, axis=0)
                transformer_std = np.sqrt(transformer.var_)
                std_check = np.allclose(
                    transformed_std, transformer_std, rtol=EPS, atol=EPS)
            else:
                std_check = True
            assert mean_check and std_check, 'The inverse transformation correctly reverts the data.'

    logger.info(
        f'Transformation using {transformer.__class__.__name__} completed.')

    if method_name not in ['fit_transform', 'fit']:
        return transformed_data
    elif method_name == 'fit_transform':
        return transformer, transformed_data
    else:
        return transformer


class _MultipleTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies a list of transformers sequentially.
    This class is for internal use and should not be instantiated directly.
    Please use the create_estimator function, or YTransformer, instead.

    Parameters
    ----------
    y_transformer : YTransformer
    """

    def __init__(self, y_transformer: YTransformer):
        logger.info('Initializing _MultipleTransformer.')
        validate_input(YTransformer, arg=y_transformer)
        self.y_transformer = y_transformer
        # self._transformers = []
        # self._is_fitted = False
        logger.info('_MultipleTransformer initialized.')

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, sanity_check: bool = False) -> "_MultipleTransformer":
        """Fit all transformers using X"""
        logger.info('Fitting all transformers in _MultipleTransformer.')
        # X = reshape_to_2d_array(X)
        # X_copy = deepcopy(X)
        if y is None:
            X = check_array(X, accept_sparse=True,
                            force_all_finite='allow-nan', ensure_2d=False)
        else:
            X, y = check_X_y(X, y, accept_sparse=True,
                             force_all_finite='allow-nan', multi_output=True)
        self.transformers_ = []
        for transformer in self.y_transformer.transformers:
            fitted_transformer = apply_transform_with_checks(
                transformer=clone(transformer.transformer), method_name='fit', X=reshape_to_2d_array(X), y=y, sanity_check=sanity_check)
            self.transformers_.append(fitted_transformer)
        logger.info('All transformers in _MultipleTransformer fitted.')
        return self

    def transform(self, X: np.ndarray, sanity_check: bool = False) -> np.ndarray:
        """Transform X using all transformers."""
        logger.info(
            'Transforming data using all transformers in _MultipleTransformer.')
        check_is_fitted(self, "transformers_")
        X = check_array(X, accept_sparse=True,
                        force_all_finite='allow-nan', ensure_2d=False)
        for transformer in self.transformers_:
            X = apply_transform_with_checks(
                transformer=transformer, method_name='transform', X=reshape_to_2d_array(X), sanity_check=sanity_check)
        logger.info(
            'Data transformation using all transformers in _MultipleTransformer completed.')
        return X

    def inverse_transform(self, X: np.ndarray, sanity_check: bool = False) -> np.ndarray:
        """Inverse transform X using all transformers."""
        logger.info(
            'Applying inverse transformation using all transformers in _MultipleTransformer.')
        check_is_fitted(self, "transformers_")
        X = check_array(X, accept_sparse=True,
                        force_all_finite='allow-nan', ensure_2d=False)
        for transformer in reversed(self.transformers_):
            X = apply_transform_with_checks(
                transformer=transformer, method_name='inverse_transform', X=X, sanity_check=sanity_check)
        logger.info(
            'Inverse transformation using all transformers in _MultipleTransformer completed.')
        return X

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        logger.info('Getting parameters for _MultipleTransformer.')
        return {"y_transformer": self.y_transformer}

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        logger.info('Setting parameters for _MultipleTransformer.')
        for parameter, value in params.items():
            setattr(self, parameter, value)
        logger.info('Parameters for _MultipleTransformer set.')
        return self


class CustomNGBRegressor(NGBRegressor):
    def __init__(self, config: ModelConfig, *args, **kwargs):
        super().__init__(Base=config.Base, Dist=config.Dist, Score=config.Score,
                         n_estimators=config.n_estimators, learning_rate=config.learning_rate,
                         col_sample=config.col_sample, minibatch_frac=config.minibatch_frac,
                         verbose=config.verbose, natural_gradient=config.natural_gradient,
                         early_stopping_rounds=config.early_stopping_rounds, *args, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        logger.info('Fitting CustomNGBRegressor.')
        X, y = check_X_y(X, y, accept_sparse=True,
                         force_all_finite='allow-nan')
        super().fit(X, y, **kwargs)
        self.fitted_ = True
        logger.info('CustomNGBRegressor fitted.')
        return self

    def predict_dist(self, X: np.ndarray):
        logger.info('Predicting distribution using CustomNGBRegressor.')
        check_is_fitted(self, "fitted_")
        X = check_array(X, accept_sparse=True, force_all_finite='allow-nan')
        y_pred_dist = super().pred_dist(X)
        logger.info('Prediction of distribution completed.')
        return y_pred_dist

    def predict(self, X: np.ndarray):
        logger.info('Predicting using CustomNGBRegressor.')
        y_pred_mean = self.predict_dist(X=X).loc
        logger.info('Prediction completed.')
        return y_pred_mean

    def predict_std(self, X: np.ndarray):
        logger.info('Predicting standard deviation using CustomNGBRegressor.')
        y_pred_std = self.predict_dist(X=X).scale
        logger.info('Prediction of standard deviation completed.')
        return y_pred_std

    @property
    def base_model(self):
        return self


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

    def validate_instance(instance, class_, default):
        if instance is not None and not isinstance(instance, class_):
            msg = f'Instance must be an instance of {class_.__name__} or None. Got {type(instance)} instead.'
            logger.error(msg)
            raise TypeError(msg)
        return instance or default()

    logger.info('Creating estimator with provided configurations.')

    model_config = validate_instance(model_config, ModelConfig, ModelConfig)
    X_transformer = validate_instance(
        X_transformer, XTransformer, XTransformer)
    y_transformer = validate_instance(
        y_transformer, YTransformer, YTransformer)

    logger.info('Building pipelines.')

    pipeline_X = Pipeline([(transformer.name, transformer.transformer)
                          for transformer in X_transformer.transformers])
    pipeline_y = _MultipleTransformer(y_transformer=y_transformer)

    pipeline_steps = [('regressor', CustomNGBRegressor(model_config))]
    if X_transformer.transformers:
        logger.info('Building feature pipeline with preprocessor.')
        pipeline_steps.insert(0, ('preprocessor', pipeline_X))

    feature_pipeline = Pipeline(pipeline_steps)

    logger.info('Building and returning CustomTransformedTargetRegressor.')

    return CustomTransformedTargetRegressor(
        regressor=feature_pipeline,
        transformer=pipeline_y
    )


class PostProcessY(BaseEstimator, TransformerMixin):
    """
    Custom transformer to postprocess data according to the specified galaxy property.

    Parameters
    ----------
    prop : GalaxyProperty
        The galaxy property to apply the inverse transform.
    """

    def __init__(self, prop: Optional[GalaxyProperty]):
        self.prop = prop
        self.inverse_transforms = self._get_label_rev_func()

    def fit(self, X: np.ndarray, y=None) -> 'PostProcessY':
        """Fit the transformer. Not used in this transformer, hence returning self."""
        return self

    @staticmethod
    def _get_label_rev_func() -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
        """Get inverse transforms for galaxy properties."""
        return {GalaxyProperty.STELLAR_MASS: lambda x: np.float_power(10, np.clip(x, a_min=0, a_max=20)),
                GalaxyProperty.DUST_MASS: lambda x: np.float_power(10, np.clip(x, a_min=0, a_max=20)) - 1,
                GalaxyProperty.METALLICITY: lambda x: np.float_power(10, np.clip(x, a_min=-1e1, a_max=1e1)),
                GalaxyProperty.SFR: lambda x: np.float_power(10, np.clip(x, a_min=0, a_max=1e2)) - 1,
                }

    def transform(self, *ys: Union[np.ndarray, Tuple[np.ndarray]]) -> Union[np.ndarray, Tuple[np.ndarray]]:
        """
        Apply inverse transformation to the data according to the specified galaxy property.

        Parameters
        ----------
        *ys : Union[np.ndarray, Tuple[np.ndarray]]
            Preprocessed data.

        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray]]
            Postprocessed data.
        """
        if self.prop is None or self.prop.value not in self.inverse_transforms.keys():
            return ys
        else:
            postprocessed_ys = [self._apply_inverse_transform(y) for y in ys]
            return tuple(postprocessed_ys) if len(postprocessed_ys) > 1 else postprocessed_ys[0]

    def _apply_inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Apply the inverse transformation for the specific galaxy property."""
        postprocessed_y = np.zeros_like(y)
        for key, func in self.inverse_transforms.items():
            if self.prop.value in key:
                postprocessed_y = func(y)
        return postprocessed_y
