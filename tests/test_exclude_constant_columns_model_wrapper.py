import numpy as np
from typing import Optional

from turbo_mc.models.matrix_completion_model import MatrixCompletionModel
from turbo_mc.models.exclude_constant_columns_model_wrapper import ExcludeConstantColumnsModelWrapper
from turbo_mc.models.matrix_factorization_model import MatrixFactorizationModel
from turbo_mc.models.column_normalizer_model_wrapper import ColumnNormalizerModelWrapper
from turbo_mc.metrics import compute_mse_global
from turbo_mc.matrix_manipulation import get_list_of_random_matrix_indices, observe_entries,\
    matrix_from_observations, complete_matrix
from turbo_mc.models.mean_model import GlobalMeanModel


def test_ColumnNormalizerModelWrapper():
    r"""
    """
    nan = np.nan
    X_observed = np.array(
        [[1.0, 1.0, 0.0, 3.0, nan, nan],
         [nan, 5.0, 0.0, 4.0, nan, 2.0],
         [1.0, nan, 0.0, 5.0, 2.0, 3.0]]
    )

    class DummyModel(MatrixCompletionModel):
        def __init__(self, n_epochs: int = 1):
            self.n_epochs = n_epochs

        def _fit_matrix_init(
                self,
                X_observed: np.array,
                Z: None = None) -> None:
            self.nrows, self.ncols = X_observed.shape

        def _step(self):
            pass

        def predict(self, r, c):
            return c
    model = ExcludeConstantColumnsModelWrapper(DummyModel())
    model.fit_matrix(X_observed)
    X_completion = model.impute_all(X_observed)
    X_completion_expected = np.array(
        [[1.0, 1.0, 0.0, 3.0, 2.0, 2.0],
         [1.0, 5.0, 0.0, 4.0, 2.0, 2.0],
         [1.0, 0.0, 0.0, 5.0, 2.0, 3.0]]
    )
    np.testing.assert_almost_equal(X_completion, X_completion_expected)


def test_ColumnNormalizerModelWrapper_2():
    r"""
    """
    nan = np.nan
    X_observed = np.array(
        [[1.0, 1.0, 0.0, 3.0, nan, nan],
         [nan, 5.0, 0.0, 4.0, nan, 2.0],
         [1.0, nan, 0.0, 5.0, 2.0, 3.0]]
    )
    model = ExcludeConstantColumnsModelWrapper(GlobalMeanModel())
    model.fit_matrix(X_observed)
    X_completion = model.impute_all(X_observed)
    X_completion_expected = np.array(
        [[1.0, 1.0, 0.0, 3.0, 2.0, 23/7],
         [1.0, 5.0, 0.0, 4.0, 2.0, 2.0],
         [1.0, 23/7, 0.0, 5.0, 2.0, 3.0]]
    )
    np.testing.assert_almost_equal(X_completion, X_completion_expected)
    X_completion_with_complete_matrix = complete_matrix(X_observed, model)
    np.testing.assert_almost_equal(X_completion, X_completion_with_complete_matrix)


def test_ColumnNormalizerModelWrapper_3():
    r"""
    Test the all-columns-constant case
    """
    nan = np.nan
    X_observed = np.array(
        [[3.0],
         [nan],
         [3.0]]
    )

    class DummyModel(MatrixCompletionModel):
        def __init__(self, n_epochs: int = 1):
            self.n_epochs = n_epochs

        def _fit_init(
                self,
                observations,
                nrows: int,
                ncols: int,
                Z: Optional[np.array] = None):
            self.nrows = nrows
            self.ncols = ncols

        def _step(self):
            pass

        def predict(self, r, c):
            return -1.0
    model = ExcludeConstantColumnsModelWrapper(DummyModel())
    model.fit_matrix(X_observed)
    X_completion = model.impute_all(X_observed)
    X_completion_expected = np.array(
        [[3.0],
         [3.0],
         [3.0]]
    )
    np.testing.assert_almost_equal(X_completion, X_completion_expected)


def test_ColumnNormalizerModelWrapper_4():
    r"""
    Create a data matrix and make a few columns zero. Using ExcludeConstantColumnsModelWrapper
    should make no difference.
    """
    np.random.seed(1)
    nrows = 19
    ncols = 20
    true_k = 1
    model_k = 1
    n_epochs = 1000
    lr_all = 0.003
    sampling_density = 0.5
    U = np.random.normal(size=(nrows, true_k))
    V = np.random.normal(size=(true_k, ncols))
    X = U @ V
    constant_cols = [0, 3, 4, 5, 18, 19]
    X[:, constant_cols] = 3.14
    random_matrix_indices = get_list_of_random_matrix_indices(nrows, ncols, sampling_density, verbose=True)
    observed_entries = observe_entries(X, random_matrix_indices, verbose=True)
    X_observed = matrix_from_observations(observed_entries, nrows=X.shape[0], ncols=X.shape[1])

    model_1 = ColumnNormalizerModelWrapper(MatrixFactorizationModel(
        n_factors=model_k, n_epochs=n_epochs, lr_all=lr_all))
    model_1.fit(observed_entries, nrows=X.shape[0], ncols=X.shape[1])
    X_completion_1 = model_1.impute_all(X_observed)

    model_2 = ExcludeConstantColumnsModelWrapper(ColumnNormalizerModelWrapper(
        MatrixFactorizationModel(n_factors=model_k, n_epochs=n_epochs, lr_all=lr_all)))
    model_2.fit(observed_entries, nrows=X.shape[0], ncols=X.shape[1])
    X_completion_2 = model_2.impute_all(X_observed)

    np.testing.assert_almost_equal(X_completion_1, X_completion_2, decimal=2)
    assert(compute_mse_global(X, X_observed, X_completion_2) < 0.03)
