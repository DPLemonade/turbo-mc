import numpy as np

from turbo_mc.models.column_normalizer_model_wrapper import ColumnNormalizerModelWrapper
from turbo_mc.models.matrix_completion_model import MatrixCompletionModel
from turbo_mc.models.mean_model import MeanModel, GlobalMeanModel
from turbo_mc.matrix_manipulation import get_observations_from_partially_observed_matrix, complete_matrix,\
    get_list_of_random_matrix_indices, observe_entries, matrix_from_observations
from turbo_mc.models.matrix_factorization_model import MatrixFactorizationModel


class DummyModel(MatrixCompletionModel):
    def __init__(self, n_epochs):
        self.n_epochs = n_epochs

    def _fit_matrix_init(
            self,
            X_observed: np.array,
            Z: None = None) -> None:
        pass

    def _step(self):
        pass

    def predict(self, r: int, c: int) -> float:
        return 0


def test_ColumnNormalizerModelWrapper_1():
    r"""
    Wrapping the GlobalMeanModel should give the mean column value as a prediction.
    """
    global_mean_model = GlobalMeanModel()
    wrapped_model = ColumnNormalizerModelWrapper(global_mean_model)
    nan = np.nan
    X = np.array(
        [[1, 2, 5, 6],
         [1, 4, 5, 6],
         [nan, nan, 5, 7],
         [nan, nan, 5, 7]]
    )
    observations = get_observations_from_partially_observed_matrix(X)
    wrapped_model.fit(observations, nrows=X.shape[0], ncols=X.shape[1])
    X_completion = complete_matrix(X, wrapped_model)
    X_expected = np.array(
        [[1, 2, 5, 6],
         [1, 4, 5, 6],
         [1, 3, 5, 7],
         [1, 3, 5, 7]]
    )
    np.testing.assert_almost_equal(X_completion, X_expected)


def test_ColumnNormalizerModelWrapper_predict_all():
    r"""
    Wrapping the GlobalMeanModel should give the mean column value as a prediction.
    """
    global_mean_model = GlobalMeanModel()
    wrapped_model = ColumnNormalizerModelWrapper(global_mean_model)
    nan = np.nan
    X = np.array(
        [[1, 2, 5, 6],
         [1, 4, 5, 6],
         [nan, nan, 5, 7],
         [nan, nan, 5, 7]]
    )
    observations = get_observations_from_partially_observed_matrix(X)
    wrapped_model.fit(observations, nrows=X.shape[0], ncols=X.shape[1])
    X_completion = X.copy()
    unobserved_entries = np.where(np.isnan(X_completion))
    X_completion[unobserved_entries] = wrapped_model.predict_all()[unobserved_entries]
    X_expected = np.array(
        [[1, 2, 5, 6],
         [1, 4, 5, 6],
         [1, 3, 5, 7],
         [1, 3, 5, 7]]
    )
    np.testing.assert_almost_equal(X_completion, X_expected)


def test_ColumnNormalizerModelWrapper_impute_all():
    r"""
    Wrapping the GlobalMeanModel should give the mean column value as a prediction.
    """
    global_mean_model = GlobalMeanModel()
    wrapped_model = ColumnNormalizerModelWrapper(global_mean_model)
    nan = np.nan
    X = np.array(
        [[1, 2, 5, 6],
         [1, 4, 5, 6],
         [nan, nan, 5, 7],
         [nan, nan, 5, 7]]
    )
    observations = get_observations_from_partially_observed_matrix(X)
    wrapped_model.fit(observations, nrows=X.shape[0], ncols=X.shape[1])
    X_completion = wrapped_model.impute_all(X)
    X_expected = np.array(
        [[1, 2, 5, 6],
         [1, 4, 5, 6],
         [1, 3, 5, 7],
         [1, 3, 5, 7]]
    )
    np.testing.assert_almost_equal(X_completion, X_expected)


def test_ColumnNormalizerModelWrapper_2():
    r"""
    The composition of ColumnNormalizerModelWrapper and GlobalMeanModel should give a MeanModel
    """
    np.random.seed(1)
    for repetition in range(10):
        R = 7
        C = 11
        X = np.random.normal(size=(R, C))
        for r in range(R):
            for c in range(C):
                if (r + c) % 3 != 0:
                    X[r, c] = np.nan
        observations = get_observations_from_partially_observed_matrix(X)
        mean_model = MeanModel()
        global_mean_model = GlobalMeanModel()
        wrapped_model = ColumnNormalizerModelWrapper(global_mean_model)
        mean_model.fit(observations, nrows=X.shape[0], ncols=X.shape[1])
        wrapped_model.fit(observations, nrows=X.shape[0], ncols=X.shape[1])
        X_completion_1 = complete_matrix(X, mean_model)
        X_completion_2 = complete_matrix(X, wrapped_model)
        np.testing.assert_almost_equal(X_completion_1, X_completion_2)


def test_ColumnNormalizerModelWrapper_predict_all_2():
    r"""
    Tests that predict_all works comparing it to a simple predict call.
    (So, if predict works, predict_all should work)
    """
    np.random.seed(1)
    R = 7
    C = 11
    X = np.random.normal(size=(R, C))
    for r in range(R):
        for c in range(C):
            if (r + c) % 3 != 0:
                X[r, c] = np.nan
    model = MatrixFactorizationModel(n_factors=2, n_epochs=10)
    wrapped_model = ColumnNormalizerModelWrapper(model)
    observations = get_observations_from_partially_observed_matrix(X)
    wrapped_model.fit(observations, nrows=X.shape[0], ncols=X.shape[1])
    X_predict_all = wrapped_model.predict_all()
    for r in range(R):
        for c in range(C):
            np.testing.assert_almost_equal(X_predict_all[r, c], wrapped_model.predict(r, c))


def test_with_perturbed_low_rank_matrix__not_wrapping_fails():
    np.random.seed(1)
    nrows = 20
    ncols = 19
    true_k = 4
    model_k = 4
    sampling_density = 0.5
    U = np.random.normal(size=(nrows, true_k))
    V = np.random.normal(size=(true_k, ncols))
    col_offsets = np.random.normal(size=(1, ncols))
    X = U @ V + col_offsets
    mf_model = MatrixFactorizationModel(n_factors=model_k, reg_all=0.0)
    random_matrix_indices = get_list_of_random_matrix_indices(nrows, ncols, sampling_density)
    observed_entries = observe_entries(X, random_matrix_indices)
    mf_model.fit(observed_entries, nrows=X.shape[0], ncols=X.shape[1])
    X_observed = matrix_from_observations(observed_entries, nrows=X.shape[0], ncols=X.shape[1])
    X_completion = complete_matrix(X_observed, mf_model)
    unobserved_indices = np.where(np.isnan(X_observed))
    error = np.array(X_completion[unobserved_indices] - X[unobserved_indices])
    mse = np.mean((error * error))
    print(f"MSE = {mse}")
    assert(mse > 1e-8)
