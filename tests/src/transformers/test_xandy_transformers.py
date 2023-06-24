import pytest
import numpy as np
from sklearn.exceptions import NotFittedError
from numpy.testing import assert_almost_equal
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays
from sklearn.preprocessing import StandardScaler, RobustScaler

from utils.reshape import reshape_to_2d_array, reshape_to_1d_array
from src.transformers.xandy_transformers import XTransformer, YTransformer, TransformerConfig


@pytest.mark.parametrize(
    "transformer_class_str, expected_mean, test_func, reshape_func",
    [('XTransformer', 0, np.allclose, lambda x: x),
     ('YTransformer', 0, np.allclose, reshape_to_1d_array)]
)
@given(arrays(float, (100, 1), elements=st.floats(-100, 100)))
@settings(deadline=None, max_examples=10)
def test_transformer_passing_and_inverse_transforming(transformer_class_str, expected_mean, test_func, reshape_func, array):
    transformer_configuration = [
        TransformerConfig(name="robust_scaler",
                          transformer=RobustScaler()),
        TransformerConfig(name="standard_scaler",
                          transformer=StandardScaler())
    ]

    if transformer_class_str == 'YTransformer':
        transformer_class = YTransformer(
            transformers=transformer_configuration)
    else:
        transformer_class = XTransformer()

    transformer = transformer_class

    # Fit and transform
    for transformer_config in transformer.transformers:
        transformed_array = transformer_config.transformer.fit_transform(
            reshape_to_2d_array(array))

        # Validate if StandardScaler is effectively applied
        if isinstance(transformer_config.transformer, StandardScaler):
            assert_almost_equal(transformed_array.mean(axis=0), expected_mean)

        # Test inverse transform right after the transformation
        inversed_array = transformer_config.transformer.inverse_transform(
            reshape_to_2d_array(transformed_array))

        assert test_func(reshape_func(inversed_array),
                         reshape_func(array), rtol=.05)


@pytest.mark.parametrize(
    "transformer_class_str",
    ['XTransformer', 'YTransformer'],
)
def test_transformer_failing(transformer_class_str):
    # Test for NotFittedError
    transformer_configuration = [
        TransformerConfig(name="robust_scaler",
                          transformer=RobustScaler()),
        TransformerConfig(name="standard_scaler",
                          transformer=StandardScaler())
    ]

    if transformer_class_str == 'YTransformer':
        transformer_class = YTransformer(
            transformers=transformer_configuration)
    else:
        transformer_class = XTransformer()

    with pytest.raises(NotFittedError):
        transformer_class.transformers[0].transformer.transform(
            np.random.rand(10, 2))


@pytest.mark.parametrize(
    "input_data",
    [["not a list"], [StandardScaler()]]
)
def test_transformer_invalid_configuration(input_data):
    with pytest.raises(ValueError):
        YTransformer(transformers=input_data)
    with pytest.raises(ValueError):
        XTransformer(transformers=input_data)


def test_empty_transformer_configuration():
    X_transformer = XTransformer(transformers=[])
    assert X_transformer.transformers == []

    X_transformer = XTransformer(transformers=None)
    assert X_transformer.transformers[0].name == "standard_scaler"
    assert isinstance(
        X_transformer.transformers[0].transformer, StandardScaler)

    y_transformer = YTransformer(transformers=[])
    assert y_transformer.transformers == []

    y_transformer = YTransformer(transformers=None)
    assert y_transformer.transformers == None
