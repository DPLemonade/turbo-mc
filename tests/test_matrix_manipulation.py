import numpy as np
import pandas as pd
import pytest
from typing import Optional

from turbo_mc.matrix_manipulation import get_observations_from_partially_observed_matrix,\
    get_list_of_random_matrix_indices, observe_entries, matrix_from_observations,\
    complete_matrix, standardize, is_constant, is_nonconstant, calc_is_nonconstant_column,\
    calc_is_constant_column, get_nonconstant_columns, get_constant_columns,\
    are_equal
from turbo_mc.models.matrix_completion_model import MatrixCompletionModel


def test_are_equal():
    assert(are_equal(
        np.array([0.0, 0.0, 0.0]),
        np.array([1e-4, 0.0, 0.0])
    ))
    assert(not are_equal(
        np.array([0.0, 0.0, 0.0]),
        np.array([1e-2, 0.0, 0.0])
    ))
    assert(not are_equal(
        np.array([0.0, 0.0, 1.0]),
        np.array([1e-4, 0.0, 0.0])
    ))
    assert(not are_equal(
        np.array([2.71, 3.14, 1.41]),
        np.array([2.71, 3.15, 1.41])
    ))
    assert(are_equal(
        np.array([2.71, 3.14, 1.41]),
        np.array([2.71, 3.14, 1.41])
    ))
    assert(are_equal(
        np.array([2.71 - 1e-4, 3.14 + 1e-4, 1.41 + 1e-4]),
        np.array([2.71 + 1e-4, 3.14 - 1e-4, 1.41 - 1e-4])
    ))


def test_constant_columns():
    nan = np.nan
    X = np.array(
        [[0.0, 1.0, nan, 1.0, nan, nan, 1e-4, 1e-2, 1e-2],
         [0.0, 1.0, nan, 2.0, 4.1, 2.0, 0, 0, nan],
         [0.0, 1.0, nan, 3.0, 2.1, nan, 0, 0, 0.0]]
    )
    is_constant_column = calc_is_constant_column(X)
    is_nonconstant_column = calc_is_nonconstant_column(X)
    constant_columns = get_constant_columns(X)
    nonconstant_columns = get_nonconstant_columns(X)

    for c in range(X.shape[1]):
        np.testing.assert_equal(is_constant_column[c], is_constant(X[:, c]))
        np.testing.assert_equal(is_nonconstant_column[c], is_nonconstant(X[:, c]))
    expected_is_constant_column = np.array([True, True, True, False, False, True, True, False, False])
    expected_is_nonconstant_column = np.array([False, False, False, True, True, False, False, True, True])
    expected_constant_columns = np.array([0, 1, 2, 5, 6])
    expected_nonconstant_columns = np.array([3, 4, 7, 8])

    np.testing.assert_equal(expected_is_constant_column, is_constant_column)
    np.testing.assert_equal(expected_is_nonconstant_column, is_nonconstant_column)
    np.testing.assert_equal(expected_constant_columns, constant_columns)
    np.testing.assert_equal(expected_nonconstant_columns, nonconstant_columns)


def test_get_observations_from_partially_observed_matrix():
    X = np.array(
        [[1.1, np.nan, 4.4],
         [2.2, 3.3, np.nan]]
    )
    observations_expected = pd.DataFrame({'row': [0, 1, 1, 0], 'col': [0, 0, 1, 2], 'val': [1.1, 2.2, 3.3, 4.4]})
    observations = get_observations_from_partially_observed_matrix(X)
    assert(np.all(observations == observations_expected))


@pytest.mark.parametrize("randomize_rows_and_columns", [False, True])
def test_get_list_of_random_matrix_indices(randomize_rows_and_columns):
    np.random.seed(1)
    res = get_list_of_random_matrix_indices(2, 2, 1.0, randomize_rows_and_columns)
    assert(set(res) == set([(0, 0), (0, 1), (1, 0), (1, 1)]))
    for repetition in range(100):
        np.random.seed(repetition)
        res = get_list_of_random_matrix_indices(2, 2, 0.6, randomize_rows_and_columns)
        assert(set(res) == set([(0, 0), (1, 1)]) or
               set(res) == set([(0, 1), (1, 0)]))
    with pytest.raises(ValueError):
        res = get_list_of_random_matrix_indices(2, 2, 0.4, randomize_rows_and_columns)


def test_observe_entries():
    X = np.array(
        [[1.1, np.nan, 4.4],
         [2.2, 3.3, np.nan]]
    )
    observations = observe_entries(X, [(0, 0), (1, 1), (0, 2), (0, 1)])
    observations_expected = pd.DataFrame({
        "row": [0, 1, 0, 0],
        "col": [0, 1, 2, 1],
        "val": [1.1, 3.3, 4.4, np.nan]
    })
    pd.testing.assert_frame_equal(observations, observations_expected)


def test_matrix_from_observations():
    observations = pd.DataFrame({
        "row": [0, 1, 0],
        "col": [0, 1, 2],
        "val": [1.1, 3.3, 4.4]
    })
    X = matrix_from_observations(observations, nrows=2, ncols=3)
    X_expected = np.array(
        [[1.1, np.nan, 4.4],
         [np.nan, 3.3, np.nan]]
    )
    np.testing.assert_almost_equal(X, X_expected)


def test_complete_matrix():
    X = np.array(
        [[1.1, np.nan, 4.4],
         [2.2, 3.3, np.nan]]
    )

    class DummyModel(MatrixCompletionModel):
        def __init__(self, n_epochs: int = 1):
            self.n_epochs = n_epochs

        def _fit_init(self, observations: pd.DataFrame, nrows: int, ncols: int, Z: Optional[np.array] = None):
            pass

        def _step(self):
            pass

        def predict(self, r: int, c: int):
            return 3.14 + r + c
    model = DummyModel()
    X_completion = complete_matrix(X, model)
    X_completion_expected = np.array(
        [[1.1, 4.14, 4.4],
         [2.2, 3.3, 6.14]]
    )
    np.testing.assert_almost_equal(X_completion, X_completion_expected)


def test_standardize_1():
    r"""
    We are specially interested in constant columns.
    """
    X = np.array(
        [[0.0, 1.0, 1.0, 2.0, -2.0, -1.0],
         [0.0, 1.0, -1.0, -2.0, 2.0, 2.0]]
    )
    X_standardized = standardize(X)
    X_standardized_expected = np.array(
        [[0.0, 0.0, 1.0, 1.0, -1.0, -1.0],
         [0.0, 0.0, -1.0, -1.0, 1.0, 1.0]]
    )
    np.testing.assert_almost_equal(X_standardized, X_standardized_expected)


def test_standardize_2():
    r"""
    We are specially interested in constant columns.
    """
    X = np.array(
        [[-10.0, 0.0, 1e16, 1e16],
         [-9.0, 0.0, 1e16, -1e16],
         [-8.0, 0.0, 1e16, 0.0]]
    )
    X_standardized = standardize(X)
    X_standardized_expected = np.array(
        [[-np.sqrt(1.5), 0.0, 0.0, np.sqrt(1.5)],
         [0.0, 0.0, 0.0, -np.sqrt(1.5)],
         [np.sqrt(1.5), 0.0, 0.0, 0.0]]
    )
    np.testing.assert_almost_equal(X_standardized, X_standardized_expected)
