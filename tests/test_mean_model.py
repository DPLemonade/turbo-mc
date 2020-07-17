import numpy as np


from turbo_mc.models.mean_model import MeanModel, GlobalMeanModel
from turbo_mc.matrix_manipulation import get_observations_from_partially_observed_matrix, complete_matrix


def test_MeanModel():
    X = np.array(
        [[1.1, np.nan, 3.2, -4.0],
         [np.nan, np.nan, 3.4, np.nan],
         [1.3, 2.14, 3.9, np.nan]]
    )
    observations = get_observations_from_partially_observed_matrix(X)
    model = MeanModel()
    model.fit(observations, nrows=X.shape[0], ncols=X.shape[1])
    X_completed = complete_matrix(X, model)
    error = (X_completed - np.array(
        [[1.1, 2.14, 3.2, -4.0],
         [1.2, 2.14, 3.4, -4.0],
         [1.3, 2.14, 3.9, -4.0]]
    ))
    assert np.mean(error * error) < 1e-8


def test_GlobalMeanModel():
    global_mean_model = GlobalMeanModel()
    nan = np.nan
    X = np.array(
        [[1, 2, 5, 6],
         [1, 4, 5, 6],
         [nan, nan, 5, 7],
         [nan, nan, 5, 7]]
    )
    observations = get_observations_from_partially_observed_matrix(X)
    global_mean_model.fit(observations, nrows=X.shape[0], ncols=X.shape[1])
    X_completion = complete_matrix(X, global_mean_model)
    X_expected = np.array(
        [[1, 2, 5, 6],
         [1, 4, 5, 6],
         [4.5, 4.5, 5, 7],
         [4.5, 4.5, 5, 7]]
    )
    np.testing.assert_almost_equal(X_completion, X_expected)
