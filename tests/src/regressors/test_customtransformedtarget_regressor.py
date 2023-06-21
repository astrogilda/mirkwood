import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline

from src.transformers.multiple_transformer import MultipleTransformer
from src.transformers.xandy_transformers import XTransformer, YTransformer, TransformerConfig
from src.regressors.customngb_regressor import ModelConfig, CustomNGBRegressor
from src.regressors.customtransformedtarget_regressor import CustomTransformedTargetRegressor, create_estimator
from utils.weightify import Weightify


@st.composite
def array_1d_and_2d(draw):
    """Strategy to generate a pair of 1D and 2D numpy arrays with shared elements"""
    n_elements = draw(st.integers(10, 100))
    unique_ratio = draw(st.floats(min_value=0.1, max_value=0.99))
    n_unique = round(n_elements * unique_ratio)
    n_repeat = n_elements - n_unique

    # feel free to adjust the range and size of this pool
    elements_pool = np.random.uniform(-100, 100, 5000)
    unique_elements = draw(st.lists(st.sampled_from(
        elements_pool), min_size=n_unique, max_size=n_unique, unique=True))

    repeat_elements = np.random.choice(unique_elements, size=n_repeat).tolist()
    elements = unique_elements + repeat_elements
    np.random.shuffle(elements)
    array_1d = np.array(elements)

    # Generate array_2d
    n_rows = array_1d.shape[0]
    n_columns = draw(st.integers(1, 50))
    array_2d = np.zeros((n_rows, n_columns + 1))
    array_2d[:, 0] = array_1d

    for i in range(n_columns):
        array_2d[:, i + 1] = np.random.permutation(array_1d)

    return array_1d, array_2d


@st.composite
def array_1d(draw):
    """Strategy to generate a 1D numpy array"""
    array_1d, _ = draw(array_1d_and_2d())
    return array_1d


@st.composite
def array_2d(draw):
    """Strategy to generate a 2D numpy array"""
    _, array_2d = draw(array_1d_and_2d())
    return array_2d


@given(array_1d_and_2d())
@settings(
    deadline=None, max_examples=10
)
def test_create_estimator(arrays):
    y, X = arrays
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42)

    model_config = ModelConfig()
    X_transformer = XTransformer()
    y_transformer = YTransformer()
    weightifier = Weightify()
    ttr = create_estimator(model_config=model_config,
                           X_transformer=X_transformer,
                           y_transformer=y_transformer,
                           weightifier=weightifier)
    assert isinstance(ttr, CustomTransformedTargetRegressor)

    ttr.fit(X_train, y_train, X_val=X_val, y_val=y_val,
            weight_flag=False, sanity_check=True)
    y_pred = ttr.predict(X)
    # Check shape of predicted y
    assert y_pred.shape == y.flatten().shape
    # Testing predict_std method
    y_pred_std = ttr.predict_std(X)
    # Check shape of predicted std
    assert y_pred_std.shape == y.flatten().shape


@given(array_1d_and_2d())
@settings(
    deadline=None, max_examples=10
)
def test_create_estimator_None(arrays):
    """Test create_estimator's response to None input"""
    # This should not raise an error and use default parameters instead
    y, X = arrays
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42)

    ttr = create_estimator(None, None, None)
    # Check if estimator is successfully created
    assert isinstance(ttr, CustomTransformedTargetRegressor)

    ttr.fit(X_train, y_train, X_val=X_val, y_val=y_val,
            weight_flag=False, sanity_check=True)
    # Check if fitting is successful
    assert ttr.regressor_ is not None
    y_pred = ttr.predict(X)
    # Check shape of predicted y
    assert y_pred.shape == y.flatten().shape
    # Testing predict_std method
    y_pred_std = ttr.predict_std(X)
    # Check shape of predicted std
    assert y_pred_std.shape == y.flatten().shape


def test_create_estimator_invalid():
    """Test create_estimator's response to invalid input"""
    with pytest.raises(TypeError):
        create_estimator("not a ModelConfig", None, None)
    with pytest.raises(TypeError):
        create_estimator(None, "not a XTransformer", None)
    with pytest.raises(TypeError):
        create_estimator(None, None, "not a YTransformer")


@given(array_1d_and_2d())
@settings(
    deadline=None, max_examples=10
)
def test_custom_transformed_target_regressor(arrays):
    """Test fitting, transforming and predicting of the CustomTransformedTargetRegressor class"""
    y, X = arrays
    X = np.nan_to_num(X)  # replace infinities with large finite numbers
    y = np.nan_to_num(y)  # replace infinities with large finite numbers

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42)

    # Instantiate transformers
    X_transformer = XTransformer()
    y_transformer = YTransformer(transformers=[TransformerConfig(
        name="robust_scaler", transformer=RobustScaler())])
    model_config = ModelConfig()

    # Instantiate the regressor
    ngb = CustomNGBRegressor(**vars(model_config))

    pipeline_X = Pipeline([(transformer.name, transformer.transformer)
                          for transformer in X_transformer.transformers])
    pipeline_y = MultipleTransformer(**vars(y_transformer))
    feature_pipeline = Pipeline([
        ('preprocessor', pipeline_X),
        ('regressor', ngb)
    ])
    ttr = CustomTransformedTargetRegressor(regressor=feature_pipeline,
                                           transformer=pipeline_y)

    # Fit and make predictions
    ttr.fit(X_train, y_train, X_val=X_val, y_val=y_val, sanity_check=True)
    y_pred = ttr.predict(X_val)

    # Fit the base regressor for comparison
    ngb.fit(X_train, y_train)
    y_pred_base = ngb.predict(X_val)

    # Predictions made by CustomTransformedTargetRegressor should be similar to those made by the base estimator
    print(f"y: {y}")
    print(f"y_pred: {y_pred}")
    print(f"y_pred_base: {y_pred_base}")
    print("\n")
    assert np.allclose(y_pred, y_pred_base, rtol=.15)
