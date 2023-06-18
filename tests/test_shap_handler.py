
@pytest.mark.parametrize("fit_before_predict", [True, False])
def test_calculate_shap_values(dummy_model_handler: ModelHandler, fit_before_predict):
    x_val = np.random.randn(100, len(FEATURE_NAMES))
    dummy_model_handler.shap_file_path = Path("test_shap_file")

    if fit_before_predict:
        dummy_model_handler.fit()

    if not fit_before_predict:
        with pytest.raises(NotFittedError):
            dummy_model_handler.calculate_shap_values(x_val)
    else:
        shap_values_val = dummy_model_handler.calculate_shap_values(x_val)
        assert isinstance(shap_values_val, np.ndarray)
        assert shap_values_val.shape == (100, len(FEATURE_NAMES))


@pytest.mark.parametrize("fit_before_predict, X_test, expected_exception", [
    (False, np.random.randn(10, len(FEATURE_NAMES)), NotFittedError),
    (True, 'invalid_input', TypeError),
    (True, np.random.randn(10, len(FEATURE_NAMES) + 1), ValueError)
])
def test_calculate_shap_values_exception(dummy_model_handler: ModelHandler, fit_before_predict, X_test, expected_exception):
    if fit_before_predict:
        dummy_model_handler.fit()

    with pytest.raises(expected_exception):
        dummy_model_handler.calculate_shap_values(X_test)


@pytest.mark.parametrize("model_type, valid_file", [
    ('estimator', True),
    ('estimator', False),
    ('shap', True),
    ('shap', False)
])
def test_save_and_load(dummy_model_handler: ModelHandler, model_type, valid_file):
    dummy_model_handler.fitting_mode = True

    if model_type == 'estimator':
        dummy_model_handler.file_path = Path("test_estimator_file")
    else:
        dummy_model_handler.shap_file_path = Path("test_shap_file")

    if not valid_file:
        dummy_model_handler.file_path = 'invalid_path'

    dummy_model_handler.fit()

    if model_type == 'shap':
        x_val = np.random.randn(100, len(FEATURE_NAMES))
        dummy_model_handler.calculate_shap_values(x_val)

    # Assert that the file was created if it's a valid file
    if valid_file:
        if model_type == 'estimator':
            assert dummy_model_handler.file_path.exists()
        else:
            assert dummy_model_handler.shap_file_path.exists()

        # Test loading from the saved file
        dummy_model_handler_loaded = ModelHandler(
            X_train=dummy_model_handler.X_train,
            y_train=dummy_model_handler.y_train,
            fitting_mode=False,
            file_path=dummy_model_handler.file_path if model_type == 'estimator' else None,
            shap_file_path=dummy_model_handler.shap_file_path if model_type == 'shap' else None,
            feature_names=dummy_model_handler.feature_names,
        )
        dummy_model_handler_loaded.fit()

        if model_type == 'shap':
            dummy_model_handler_loaded.calculate_shap_values(x_val)

        # Check if the loaded model has been fit
        assert dummy_model_handler_loaded._estimator_handler.is_fitted == True

    # Delete the file after the test if it's valid
    if valid_file:
        if model_type == 'estimator':
            os.remove(dummy_model_handler.file_path)
        else:
            os.remove(dummy_model_handler.shap_file_path)
