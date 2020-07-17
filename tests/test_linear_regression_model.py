import numpy as np
import pytest

from turbo_mc.matrix_manipulation import get_list_of_random_matrix_indices,\
    observe_entries, matrix_from_observations
from turbo_mc.models.linear_regression_model import LinearRegressionModel


def test_linear_regression_unknown_solver_raises():
    r"""
    Initializing LinearRegressionModel with an unknown solver should raise.
    """
    with pytest.raises(ValueError):
        LinearRegressionModel(solver='Unexistent solver')


def test_linear_regression_fitting_intercept_with_manual_solver_fails():
    r"""
    The manual solver does not support fitting the intercept. Should raise.
    """
    with pytest.raises(ValueError):
        LinearRegressionModel(fit_intercept=True, solver='manual')


@pytest.mark.parametrize(
    "solver",
    [('sklearn'),
     ('manual')]
    )
def test_linear_regression_model_basic(solver):
    r"""
    Basic test for the LinearRegressionModel
    """
    R = 11
    C = 100
    K = 3
    np.random.seed(1)
    Z = np.random.uniform(size=(R, K))
    W = np.random.uniform(size=(K, C))
    X_true = Z @ W
    # nan-out the last row of the matrix to see that even without observations we can recover it.
    # For each column, exactly 3 rows will be observed, which is exactly what we need to
    # recover the column embeddings.
    sampling_density = 0.3
    random_matrix_indices = get_list_of_random_matrix_indices(R - 1, C, sampling_density)
    observed_entries = observe_entries(X_true, random_matrix_indices, verbose=True)
    X_observed = matrix_from_observations(observed_entries, nrows=X_true.shape[0], ncols=X_true.shape[1])
    print(X_observed)
    model = LinearRegressionModel(solver=solver)
    model.fit_matrix(X_observed, Z)
    X_completion = model.predict_all()
    np.testing.assert_almost_equal(X_true, X_completion)


@pytest.mark.parametrize(
    "solver",
    [('sklearn'),
     ('manual')]
    )
def test_linear_regression_model_medium(solver):
    r"""
    Medium-sized test for the LinearRegressionModel (more rows, larger K)
    """
    R = 290
    C = 100
    K = 32
    np.random.seed(1)
    Z = np.random.uniform(size=(R, K))
    W = np.random.uniform(size=(K, C))
    X_true = Z @ W
    sampling_density = 0.2
    random_matrix_indices = get_list_of_random_matrix_indices(R - 1, C, sampling_density)
    observed_entries = observe_entries(X_true, random_matrix_indices, verbose=True)
    X_observed = matrix_from_observations(observed_entries, nrows=X_true.shape[0], ncols=X_true.shape[1])
    print(X_observed)
    model = LinearRegressionModel(solver=solver)
    model.fit_matrix(X_observed, Z)
    X_completion = model.predict_all()
    np.testing.assert_almost_equal(X_true, X_completion)


@pytest.mark.parametrize(
    "fit_intercept,add_intercept,should_raise",
    [(False, False, False),
     (False, True, True),
     (True, True, False),
     (True, False, False)]
    )
def test_linear_regression_model_basic_intercept(
        fit_intercept: bool,
        add_intercept: bool,
        should_raise: bool):
    r"""
    Tests that fitting the intercept works exactly when it should.
    """
    R = 40
    C = 10
    K = 5
    np.random.seed(1)
    Z = np.random.uniform(size=(R, K))
    W = np.random.uniform(size=(K, C))
    X_true = Z @ W
    if add_intercept:
        X_true += np.random.uniform(size=(1, C))
    sampling_density = 0.5
    random_matrix_indices = get_list_of_random_matrix_indices(R - 1, C, sampling_density)
    observed_entries = observe_entries(X_true, random_matrix_indices, verbose=True)
    X_observed = matrix_from_observations(observed_entries, nrows=X_true.shape[0], ncols=X_true.shape[1])
    print(X_observed)
    model = LinearRegressionModel(fit_intercept=fit_intercept)
    model.fit_matrix(X_observed, Z)
    X_completion = model.predict_all()
    if not should_raise:
        np.testing.assert_almost_equal(X_true, X_completion)
    else:
        with pytest.raises(AssertionError):
            np.testing.assert_almost_equal(X_true, X_completion)

    # Now check that 'predict' works
    X_completion_with_predict = np.zeros_like(X_true, dtype=float)
    for r in range(R):
        for c in range(C):
            X_completion_with_predict[r, c] = model.predict(r, c)
    np.testing.assert_almost_equal(X_completion, X_completion_with_predict)
