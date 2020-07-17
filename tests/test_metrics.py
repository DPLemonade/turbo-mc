import numpy as np
import pytest

from turbo_mc.matrix_manipulation import standardize, get_nonconstant_columns
from turbo_mc.metrics import compute_r2s, compute_mse_global, compute_mse_per_column,\
    compute_mse_giving_equal_weight_to_all_columns, compute_variance_explained,\
    variance_explained_upper_bounds, compute_best_low_rank_approximation,\
    compute_pearson_r2s, compute_spearman_r2s
from turbo_mc.models.exclude_constant_columns_model_wrapper import ExcludeConstantColumnsModelWrapper
from turbo_mc.models.column_normalizer_model_wrapper import ColumnNormalizerModelWrapper
from turbo_mc.models.matrix_factorization_model import MatrixFactorizationModel


def standardize_jointly(X: np.array, X_pred: np.array) -> np.array:
    r"""
    Constant columns in X are standardized to zero in both X and X_pred, even if X_pred
    is non-zero there! For all other columns we subtract the mean of X and divide by the std.
    """
    X_pred = X_pred.copy()
    X = X.copy()
    column_stds = np.std(X, axis=0).flatten()
    nonconstant_columns = get_nonconstant_columns(X)
    X_nonconstant = X[:, nonconstant_columns]
    column_means = np.expand_dims(np.mean(X_nonconstant, axis=0).flatten(), axis=0)
    column_stds = np.expand_dims(np.std(X_nonconstant, axis=0).flatten(), axis=0)
    X_nonconstant_standardized = (X_nonconstant - column_means) / column_stds
    X_pred_nonconstant = X_pred[:, nonconstant_columns]
    X_pred_nonconstant_standardized = (X_pred_nonconstant - column_means) / column_stds
    X_standardized = np.zeros_like(X)
    X_standardized[:, nonconstant_columns] = X_nonconstant_standardized
    X_pred_standardized = np.zeros_like(X_pred)
    X_pred_standardized[:, nonconstant_columns] = X_pred_nonconstant_standardized
    return X_standardized, X_pred_standardized


def test_variance_explained_upper_bound_is_satiated_basic():
    r"""
    the MF model should satiate the variance explained upper bound.
    """
    np.random.seed(1)
    R = 19
    C = 20
    true_k = 2
    model_k = 1
    give_each_column_equal_weight = False
    U = np.random.normal(size=(R, true_k))
    V = np.random.normal(size=(true_k, C))
    X = U @ V
    model = \
        MatrixFactorizationModel(
            n_factors=model_k,
            n_epochs=1000,
            reg_all=0.0,
            lr_all=0.03,
            biased=False
        )
    model.fit_matrix(X)
    X_predicted = model.predict_all()
    variance_explained = compute_variance_explained(
        X, X_predicted, give_each_column_equal_weight)
    upper_bounds = variance_explained_upper_bounds(X, give_each_column_equal_weight)

    X_best_low_rank = compute_best_low_rank_approximation(X, model_k)
    variance_explained_by_best_low_rank_approx = \
        1.0 - np.sum((X - X_best_low_rank) ** 2) / np.sum(X * X)
    relative_error_to_best_low_rank_approx =\
        np.sum((X_predicted - X_best_low_rank) ** 2) / np.sum(X * X)
    np.testing.assert_almost_equal(relative_error_to_best_low_rank_approx, 0.0, decimal=3)

    # plt.plot(upper_bounds)
    print(f"%.2f <=> %.2f" % (variance_explained, upper_bounds[model_k - 1]))
    np.testing.assert_almost_equal(variance_explained, upper_bounds[model_k - 1], decimal=2)
    np.testing.assert_almost_equal(variance_explained, variance_explained_by_best_low_rank_approx, decimal=2)


def test_variance_explained_upper_bound_is_satiated_equal_weight_basic():
    r"""
    the MF model should satiate the variance explained upper bound.
    """
    np.random.seed(1)
    R = 19
    C = 20
    true_k = 2
    model_k = 1
    give_each_column_equal_weight = True
    U = np.random.normal(size=(R, true_k))
    V = np.random.normal(size=(true_k, C))
    X = U @ V
    model = \
        ColumnNormalizerModelWrapper(
            MatrixFactorizationModel(
                n_factors=model_k,
                n_epochs=1000,
                reg_all=0.0,
                lr_all=0.01,
                biased=False
            ))
    model.fit_matrix(X)
    X_predicted = model.predict_all()
    variance_explained = compute_variance_explained(
        X, X_predicted, give_each_column_equal_weight)
    upper_bounds = variance_explained_upper_bounds(X, give_each_column_equal_weight)

    X_standardized = standardize(X)
    X_best_low_rank = compute_best_low_rank_approximation(X_standardized, model_k)
    variance_explained_by_best_low_rank_approx = \
        1.0 - np.sum((X_standardized - X_best_low_rank) ** 2) / np.sum(X_standardized * X_standardized)
    _, X_predicted_standardized = standardize_jointly(X, X_predicted)
    relative_error_to_best_low_rank_approx =\
        np.sum((X_predicted_standardized - X_best_low_rank) ** 2) / np.sum(X_best_low_rank * X_best_low_rank)
    np.testing.assert_almost_equal(relative_error_to_best_low_rank_approx, 0.0, decimal=2)

    print(f"%.2f <=> %.2f" % (variance_explained, upper_bounds[model_k - 1]))
    np.testing.assert_almost_equal(variance_explained, upper_bounds[model_k - 1], decimal=2)
    np.testing.assert_almost_equal(variance_explained, variance_explained_by_best_low_rank_approx, decimal=2)


def test_variance_explained_upper_bound_is_satiated_equal_weight_constant_cols_basic():
    r"""
    the MF model should satiate the variance explained upper bound.
    """
    np.random.seed(1)
    R = 19
    C = 20
    true_k = 2
    model_k = 1
    give_each_column_equal_weight = True
    U = np.random.normal(size=(R, true_k))
    V = np.random.normal(size=(true_k, C))
    X = U @ V
    constant_columns = [0, 3, 7, 8, 9, 18, 19]
    X[:, constant_columns] = 3.14
    model = \
        ExcludeConstantColumnsModelWrapper(  # This _should not_ have an effect.
            ColumnNormalizerModelWrapper(
                MatrixFactorizationModel(
                    n_factors=model_k,
                    n_epochs=1000,
                    reg_all=0.0,
                    lr_all=0.01,
                    biased=False
                )))
    model.fit_matrix(X)
    X_predicted = model.predict_all()
    variance_explained = compute_variance_explained(
        X, X_predicted, give_each_column_equal_weight)
    upper_bounds = variance_explained_upper_bounds(X, give_each_column_equal_weight)

    X_standardized = standardize(X)
    X_best_low_rank = compute_best_low_rank_approximation(X_standardized, model_k)
    variance_explained_by_best_low_rank_approx = \
        1.0 - np.sum((X_standardized - X_best_low_rank) ** 2) / np.sum(X_standardized * X_standardized)
    _, X_predicted_standardized = standardize_jointly(X, X_predicted)
    relative_error_to_best_low_rank_approx =\
        np.sum((X_predicted_standardized - X_best_low_rank) ** 2) / np.sum(X_best_low_rank * X_best_low_rank)
    np.testing.assert_almost_equal(relative_error_to_best_low_rank_approx, 0.0, decimal=2)

    print(f"%.2f <=> %.2f" % (variance_explained, upper_bounds[model_k - 1]))
    np.testing.assert_almost_equal(variance_explained, upper_bounds[model_k - 1], decimal=2)
    np.testing.assert_almost_equal(variance_explained, variance_explained_by_best_low_rank_approx, decimal=2)


def test_variance_explained_upper_bound_is_satiated_full():
    r"""
    the MF model should satiate the variance explained upper bound.
    """
    np.random.seed(1)
    R = 19
    C = 20
    true_k = 14
    give_each_column_equal_weight = False
    U = np.random.normal(size=(R, true_k))
    V = np.random.normal(size=(true_k, C))
    X = U @ V
    for model_k in range(1, true_k + 1, 1):
        print(f"model_k = {model_k}")
        model = MatrixFactorizationModel(
            n_factors=model_k,
            n_epochs=100,
            reg_all=0.0,
            lr_all=0.01,
            biased=False
        )
        model.fit_matrix(X)
        X_predicted = model.predict_all()
        variance_explained = compute_variance_explained(
            X, X_predicted, give_each_column_equal_weight)
        upper_bounds = variance_explained_upper_bounds(X, give_each_column_equal_weight)

        X_best_low_rank = compute_best_low_rank_approximation(X, model_k)
        variance_explained_by_best_low_rank_approx = \
            1.0 - np.sum((X - X_best_low_rank) ** 2) / np.sum(X * X)
        relative_error_to_best_low_rank_approx =\
            np.sum((X_predicted - X_best_low_rank) ** 2) / np.sum(X * X)
        np.testing.assert_almost_equal(relative_error_to_best_low_rank_approx, 0.0, decimal=1)

        # plt.plot(upper_bounds)
        print(f"%.2f <=> %.2f" % (variance_explained, upper_bounds[model_k - 1]))
        np.testing.assert_almost_equal(variance_explained, upper_bounds[model_k - 1], decimal=2)
        np.testing.assert_almost_equal(variance_explained, variance_explained_by_best_low_rank_approx, decimal=2)


@pytest.mark.slow
def test_variance_explained_upper_bound_is_satiated_equal_weight_full():
    r"""
    the MF model should satiate the variance explained upper bound.
    """
    np.random.seed(1)
    R = 19
    C = 20
    true_k = 14
    give_each_column_equal_weight = True
    U = np.random.normal(size=(R, true_k))
    V = np.random.normal(size=(true_k, C))
    X = U @ V
    for model_k in range(1, true_k + 1, 1):
        model = \
            ColumnNormalizerModelWrapper(
                MatrixFactorizationModel(
                    n_factors=model_k,
                    n_epochs=2000,
                    reg_all=0.0,
                    lr_all=0.01,
                    biased=False
                ))
        model.fit_matrix(X)
        X_predicted = model.predict_all()
        variance_explained = compute_variance_explained(
            X, X_predicted, give_each_column_equal_weight)
        upper_bounds = variance_explained_upper_bounds(X, give_each_column_equal_weight)

        X_standardized = standardize(X)
        X_best_low_rank = compute_best_low_rank_approximation(X_standardized, model_k)
        variance_explained_by_best_low_rank_approx = \
            1.0 - np.sum((X_standardized - X_best_low_rank) ** 2) / np.sum(X_standardized * X_standardized)
        _, X_predicted_standardized = standardize_jointly(X, X_predicted)
        relative_error_to_best_low_rank_approx =\
            np.sum((X_predicted_standardized - X_best_low_rank) ** 2) / np.sum(X_best_low_rank * X_best_low_rank)
        np.testing.assert_almost_equal(relative_error_to_best_low_rank_approx, 0.0, decimal=2)

        print(f"%.2f <=> %.2f" % (variance_explained, upper_bounds[model_k - 1]))
        np.testing.assert_almost_equal(variance_explained, upper_bounds[model_k - 1], decimal=2)
        np.testing.assert_almost_equal(variance_explained, variance_explained_by_best_low_rank_approx, decimal=2)


@pytest.mark.slow
def test_variance_explained_upper_bound_is_satiated_equal_weight_constant_cols_full():
    r"""
    the MF model should satiate the variance explained upper bound.
    """
    np.random.seed(1)
    R = 19
    C = 20
    true_k = 14
    give_each_column_equal_weight = True
    U = np.random.normal(size=(R, true_k))
    V = np.random.normal(size=(true_k, C))
    X = U @ V
    constant_columns = [0, 7, 8, 9, 18, 19]
    X[:, constant_columns] = 3.14
    for model_k in range(1, true_k + 1, 1):
        model = \
            ExcludeConstantColumnsModelWrapper(  # This _should not_ have an effect.
                ColumnNormalizerModelWrapper(
                    MatrixFactorizationModel(
                        n_factors=model_k,
                        n_epochs=2000,
                        reg_all=0.0,
                        lr_all=0.01,
                        biased=False
                    )))
        model.fit_matrix(X)
        X_predicted = model.predict_all()
        variance_explained = compute_variance_explained(
            X, X_predicted, give_each_column_equal_weight)
        upper_bounds = variance_explained_upper_bounds(X, give_each_column_equal_weight)

        X_standardized = standardize(X)
        X_best_low_rank = compute_best_low_rank_approximation(X_standardized, model_k)
        variance_explained_by_best_low_rank_approx = \
            1.0 - np.sum((X_standardized - X_best_low_rank) ** 2) / np.sum(X_standardized * X_standardized)
        _, X_predicted_standardized = standardize_jointly(X, X_predicted)
        relative_error_to_best_low_rank_approx =\
            np.sum((X_predicted_standardized - X_best_low_rank) ** 2) / np.sum(X_best_low_rank * X_best_low_rank)
        np.testing.assert_almost_equal(relative_error_to_best_low_rank_approx, 0.0, decimal=2)

        print(f"%.2f <=> %.2f" % (variance_explained, upper_bounds[model_k - 1]))
        np.testing.assert_almost_equal(variance_explained, upper_bounds[model_k - 1], decimal=2)
        np.testing.assert_almost_equal(variance_explained, variance_explained_by_best_low_rank_approx, decimal=2)


def test_compute_r2s():
    r"""
    Test that r-squared is being computed correctly.
    """
    nan = np.nan
    X = np.array(
        [[+3.0, 2, 2, 2, 1, 1, 6],
         [-0.5, 2, 1, 1, 1, 1, 6],
         [+2.0, 2, 2, 1, 1, 1, 6],
         [+7.0, 2, 2, 1, 1, 1, 6],
         [+3.0, 1, 1, 3, 1, 1, 5],
         [-0.5, 2, 3, 2, 1, 1, 5],
         [+2.0, 2, 3, 3, 1, 1, 5],
         [+7.0, 3, 2, 3, 1, 1, 5]]
    )
    X_observed = np.array(
        [[nan, 2, 2, nan, nan, nan, nan],
         [nan, 2, nan, 1, 1, 1, nan],
         [nan, 2, 2, nan, 1, 1, nan],
         [nan, 2, nan, 1, 1, 1, nan],
         [+3.0, nan, 1, 3, 1, 1, 5],
         [-0.5, nan, 3, 2, 1, 1, 5],
         [+2.0, 2, nan, 3, 1, 1, 5],
         [+7.0, nan, 2, nan, 1, 1, 5]]
    )
    X_completion = np.array(
        [[+2.5, 2, 2, 2, 1, 2, 6],
         [+0.0, 2, 2, 1, 1, 1, 6],
         [+2.0, 2, 2, 3, 1, 1, 5],
         [+8.0, 2, 2, 1, 1, 1, 5],
         [+3.0, 1, 1, 3, 1, 1, 5],
         [-0.5, 2, 3, 2, 1, 1, 5],
         [+2.0, 2, 2, 3, 1, 1, 5],
         [+7.0, 3, 2, 1, 1, 1, 5]]
    )
    r2s = compute_r2s(X, X_observed, X_completion)
    expected_r2s = np.array([0.9486081, 1.0, 0.0, -1.0, 1.0, -1.0, 0.5])
    np.testing.assert_almost_equal(r2s, expected_r2s)


def test_compute_mse_per_column():
    r"""
    Test MSE per column is computed correctly, both when including and when excluding
    observed entries. Also check that returning SSE instead of MSE works.
    When give_each_column_equal_weight=True, MSE should be zero if predictions are close enough,
    else should be nan.
    """
    nan = np.nan
    X = np.array(
        [[1.0, 5.0, 9.0, 1.0, 0.0, 0.0, 0.0],
         [2.0, 6.0, 10.0, 1.0, 0.0, 0.0, 0.0],
         [3.0, 7.0, 11.0, 1.0, 0.0, 0.0, 0.0],
         [4.0, 8.0, 12.0, 1.0, 0.0, 0.0, 0.0]]
    )
    X_observed = np.array(
        [[1.0, 5.0, nan, nan, nan, nan, nan],
         [nan, 6.0, nan, nan, nan, nan, nan],
         [nan, 7.0, 11.0, 1.0, 0.0, 0.0, 0.0],
         [4.0, 8.0, nan, 1.0, 0.0, 0.0, 0.0]]
    )
    X_completion = np.array(
        [[1.0, 5.0, 7.0, 1.0, 1.0, 1e-16, 1e-3],
         [4.0, 6.0, 11.0, 1.0, 0.0, -1e-16, 0.0],
         [3.0, 7.0, 11.0, 1.0, 0.0, 1e-16, 0.0],
         [4.0, 8.0, 10.0, 1.0, 0.0, -1e-16, 0.0]]
    )
    # With exclude_observed=True, which is the default
    mse_per_column = compute_mse_per_column(X, X_observed, X_completion)
    mse_per_column_expected = np.array([2.0, 0.0, 3.0, 0.0, 0.5, 0.0, 1e-6 / 2.0])
    np.testing.assert_almost_equal(mse_per_column, mse_per_column_expected)

    sse_per_column = compute_mse_per_column(X, X_observed, X_completion, return_sum_instead_of_mean=True)
    sse_per_column_expected = np.array([4.0, 0.0, 9.0, 0.0, 1.0, 0.0, 2e-6 / 2.0])
    np.testing.assert_almost_equal(sse_per_column, sse_per_column_expected)

    # With exclude_observed=False
    mse_per_column = compute_mse_per_column(X, X_observed, X_completion, exclude_observed=False)
    mse_per_column_expected = np.array([1.0, 0.0, 2.25, 0.0, 0.25, 0.0, 1e-6 / 4.0])
    np.testing.assert_almost_equal(mse_per_column, mse_per_column_expected)

    sse_per_column = compute_mse_per_column(X, X_observed, X_completion, exclude_observed=False,
                                            return_sum_instead_of_mean=True)
    sse_per_column_expected = np.array([4.0, 0.0, 9.0, 0.0, 1.0, 0.0, 4 * 1e-6 / 4.0])
    np.testing.assert_almost_equal(sse_per_column, sse_per_column_expected)

    # Repeat above normalizing MSE by column std
    mse_per_column = compute_mse_per_column(X, X_observed, X_completion,
                                            give_each_column_equal_weight=True)
    mse_per_column_expected = np.array([1.6, 0.0, 2.4, 0.0, nan, 0.0, nan])
    np.testing.assert_almost_equal(mse_per_column, mse_per_column_expected)

    sse_per_column = compute_mse_per_column(X, X_observed, X_completion,
                                            give_each_column_equal_weight=True,
                                            return_sum_instead_of_mean=True)
    sse_per_column_expected = np.array([3.2, 0.0, 7.2, 0.0, nan, 0.0, nan])
    np.testing.assert_almost_equal(sse_per_column, sse_per_column_expected)

    # Scaling columns shouldn't change result
    column_multiplers = np.array([[3.14, 2.71, 2.0, 3.0, 5.0, 2.0, 2.0]])
    mse_per_column = compute_mse_per_column(X * column_multiplers, X_observed * column_multiplers,
                                            X_completion * column_multiplers,
                                            give_each_column_equal_weight=True)
    mse_per_column_expected = np.array([1.6, 0.0, 2.4, 0.0, nan, 0.0, nan])  # This is a fun border case.
    np.testing.assert_almost_equal(mse_per_column, mse_per_column_expected)

    sse_per_column = compute_mse_per_column(X * column_multiplers, X_observed * column_multiplers,
                                            X_completion * column_multiplers,
                                            give_each_column_equal_weight=True,
                                            return_sum_instead_of_mean=True)
    sse_per_column_expected = np.array([3.2, 0.0, 7.2, 0.0, nan, 0.0, nan])  # This is a fun border case.
    np.testing.assert_almost_equal(sse_per_column, sse_per_column_expected)

    # Scaling rows should break results
    row_multiplers = np.array([[3.14], [2.71], [2.0], [3.0]])
    mse_per_column = compute_mse_per_column(X * row_multiplers, X_observed * row_multiplers,
                                            X_completion * row_multiplers,
                                            give_each_column_equal_weight=True)
    mse_per_column_expected = np.array([1.6, 0.0, 2.4, 0.0, nan, 0.0, nan])
    with pytest.raises(AssertionError):
        np.testing.assert_almost_equal(mse_per_column, mse_per_column_expected)

    sse_per_column = compute_mse_per_column(X * row_multiplers, X_observed * row_multiplers,
                                            X_completion * row_multiplers,
                                            give_each_column_equal_weight=True,
                                            return_sum_instead_of_mean=True)
    sse_per_column_expected = np.array([3.2, 0.0, 7.2, 0.0, nan, 0.0, nan])
    with pytest.raises(AssertionError):
        np.testing.assert_almost_equal(sse_per_column, sse_per_column_expected)

    # Now we're back to excluding observed entries
    mse_per_column = compute_mse_per_column(X, X_observed, X_completion, exclude_observed=False,
                                            give_each_column_equal_weight=True)
    mse_per_column_expected = np.array([0.8, 0.0, 1.8, 0.0, nan, 0.0, nan])
    np.testing.assert_almost_equal(mse_per_column, mse_per_column_expected)

    sse_per_column = compute_mse_per_column(X, X_observed, X_completion, exclude_observed=False,
                                            give_each_column_equal_weight=True,
                                            return_sum_instead_of_mean=True)
    sse_per_column_expected = np.array([3.2, 0.0, 7.2, 0.0, nan, 0.0, nan])
    np.testing.assert_almost_equal(sse_per_column, sse_per_column_expected)

    # Scaling columns shouldn't change result
    column_multiplers = np.array([[3.14, 2.71, 2.0, 3.0, 5.0, 2.0, 2.0]])
    mse_per_column = compute_mse_per_column(X * column_multiplers, X_observed * column_multiplers,
                                            X_completion * column_multiplers, exclude_observed=False,
                                            give_each_column_equal_weight=True)
    mse_per_column_expected = np.array([0.8, 0.0, 1.8, 0.0, nan, 0.0, nan])
    np.testing.assert_almost_equal(mse_per_column, mse_per_column_expected)

    sse_per_column = compute_mse_per_column(X * column_multiplers, X_observed * column_multiplers,
                                            X_completion * column_multiplers, exclude_observed=False,
                                            give_each_column_equal_weight=True,
                                            return_sum_instead_of_mean=True)
    sse_per_column_expected = np.array([3.2, 0.0, 7.2, 0.0, nan, 0.0, nan])
    np.testing.assert_almost_equal(sse_per_column, sse_per_column_expected)

    # Scaling rows should break results
    row_multiplers = np.array([[3.14], [2.71], [2.0], [3.0]])
    mse_per_column = compute_mse_per_column(X * row_multiplers, X_observed * row_multiplers,
                                            X_completion * row_multiplers, exclude_observed=False,
                                            give_each_column_equal_weight=True)
    mse_per_column_expected = np.array([0.8, 0.0, 1.8, 0.0, nan, 0.0, nan])
    with pytest.raises(AssertionError):
        np.testing.assert_almost_equal(mse_per_column, mse_per_column_expected)

    sse_per_column = compute_mse_per_column(X * row_multiplers, X_observed * row_multiplers,
                                            X_completion * row_multiplers, exclude_observed=False,
                                            give_each_column_equal_weight=True)
    sse_per_column_expected = np.array([3.2, 0.0, 7.2, 0.0, nan, 0.0, nan])
    with pytest.raises(AssertionError):
        np.testing.assert_almost_equal(sse_per_column, sse_per_column_expected)


def test_compute_mse_giving_equal_weight_to_all_columns_2():
    X = np.array(
        [[0.0]]
    )
    X_observed = np.array(
        [[0.0]]
    )
    X_completion = np.array(
        [[0.0]]
    )
    np.testing.assert_almost_equal(compute_mse_giving_equal_weight_to_all_columns(
        X, X_observed, X_completion
    ), 0.0)
    np.testing.assert_almost_equal(compute_mse_giving_equal_weight_to_all_columns(
        X, X_observed, X_completion, exclude_observed=False
    ), 0.0)


def test_compute_mse_giving_equal_weight_to_all_columns_3():
    nan = np.nan
    X = np.array(
        [[0.0]]
    )
    X_observed = np.array(
        [[nan]]
    )
    X_completion = np.array(
        [[1.0]]
    )
    np.testing.assert_almost_equal(compute_mse_giving_equal_weight_to_all_columns(
        X, X_observed, X_completion
    ), nan)
    np.testing.assert_almost_equal(compute_mse_giving_equal_weight_to_all_columns(
        X, X_observed, X_completion, exclude_observed=False
    ), nan)


def test_compute_mse_giving_equal_weight_to_all_columns_4():
    nan = np.nan
    X = np.array(
        [[0.0, 0.0]]
    )
    X_observed = np.array(
        [[nan, nan]]
    )
    X_completion = np.array(
        [[1.0, 0.0]]
    )
    np.testing.assert_almost_equal(compute_mse_giving_equal_weight_to_all_columns(
        X, X_observed, X_completion
    ), nan)
    np.testing.assert_almost_equal(compute_mse_giving_equal_weight_to_all_columns(
        X, X_observed, X_completion, exclude_observed=False
    ), nan)


def test_compute_mse_giving_equal_weight_to_all_columns_5():
    nan = np.nan
    X = np.array(
        [[0.0]]
    )
    X_observed = np.array(
        [[nan]]
    )
    X_completion = np.array(
        [[0.0]]
    )
    np.testing.assert_almost_equal(compute_mse_giving_equal_weight_to_all_columns(
        X, X_observed, X_completion
    ), 0.0)
    np.testing.assert_almost_equal(compute_mse_giving_equal_weight_to_all_columns(
        X, X_observed, X_completion, exclude_observed=False
    ), 0.0)


def test_compute_mse_giving_equal_weight_to_all_columns_6():
    X = np.array(
        [[0.0, 0.0]]
    )
    X_observed = np.array(
        [[0.0, 0.0]]
    )
    X_completion = np.array(
        [[1.0, 0.0]]
    )
    np.testing.assert_almost_equal(compute_mse_giving_equal_weight_to_all_columns(
        X, X_observed, X_completion
    ), 0.0)
    np.testing.assert_almost_equal(compute_mse_giving_equal_weight_to_all_columns(
        X, X_observed, X_completion, exclude_observed=False
    ), np.nan)


def test_compute_mse_giving_equal_weight_to_all_columns():
    r"""
    Test that columns are effectively equally weighted and that np.nans are correctly ignored.
    Third and fourth features are constant.
    """
    nan = np.nan
    X = np.array(
        [[10.0, 4.0, 1.0, 0.0, 3.0, 3.0],
         [20.0, 5.0, 1.0, 0.0, 4.0, 3.0],
         [30.0, 6.0, 1.0, 0.0, 5.0, 3.0]]
    )
    X_observed = np.array(
        [[nan, 4.0, 1.0, nan, 3.0, 3.0],
         [20., 5.0, 1.0, 0.0, 4.0, 3.0],
         [nan, nan, nan, 0.0, 5.0, 3.0]]
    )
    X_completion = np.array(
        [[70.0, 4.0, 1.0, 0.0, 3.0, 3.0],
         [20.0, 5.0, 1.0, 0.0, 4.0, 3.0],
         [100., 9.0, 1.0, 0.0, 5.0, 3.0]]
    )
    mse_giving_equal_weight_to_all_columns = compute_mse_giving_equal_weight_to_all_columns(
        X, X_observed, X_completion)
    mse_giving_equal_weight_to_all_columns_expected = (54 + 73.5 + 13.5 + 0.0 + 0.0) / 5.0
    np.testing.assert_almost_equal(
        mse_giving_equal_weight_to_all_columns,
        mse_giving_equal_weight_to_all_columns_expected
    )

    mse_giving_equal_weight_to_all_columns = compute_mse_giving_equal_weight_to_all_columns(
        X, X_observed, X_completion, exclude_observed=False)
    mse_giving_equal_weight_to_all_columns_expected = \
        (54 + 0.0 + 73.5 + 0.0 + 0.0 + 13.5 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0
         + 0.0 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0) / 18.0
    np.testing.assert_almost_equal(
        mse_giving_equal_weight_to_all_columns,
        mse_giving_equal_weight_to_all_columns_expected
    )


def test_compute_mse_global():
    r"""
    Test global MSE is computed correctly, both when including and when excluding
    observed entries.
    """
    nan = np.nan
    X = np.array(
        [[1.0, 3.0, 5.0, 7.0],
         [2.0, 4.0, 6.0, 8.0]]
    )
    X_observed = np.array(
        [[nan, 3.0, nan, 7.0],
         [2.0, nan, 6.0, 8.0]]
    )
    X_completion = np.array(
        [[1.0, 3.0, 3.0, 7.0],
         [3.0, 2.0, 6.0, 8.0]]
    )
    mse = compute_mse_global(X, X_observed, X_completion)
    mse_expected = 8.0/3.0
    np.testing.assert_almost_equal(mse, mse_expected)
    mse = compute_mse_global(X, X_observed, X_completion, exclude_observed=False)
    mse_expected = 9.0/8.0
    np.testing.assert_almost_equal(mse, mse_expected)


def test_compute_mse_global_border_case():
    r"""
    When all entries are observed, should return 0 even if we exclude observed entries.
    """
    X = np.array(
        [[1.0, 3.0, 5.0, 7.0],
         [2.0, 4.0, 6.0, 8.0]]
    )
    X_observed = X
    X_completion = X
    mse = compute_mse_global(X, X_observed, X_completion)
    mse_expected = 0.0
    np.testing.assert_almost_equal(mse, mse_expected)
    mse = compute_mse_global(X, X_observed, X_completion, exclude_observed=False)
    mse_expected = 0.0
    np.testing.assert_almost_equal(mse, mse_expected)


def test_compute_variance_explained_and_compute_mse_per_column_agree():
    r"""
    These two independent ways of computing the variance explained should agree.
    This is a regression test that makes sure my assumptions about how nonconstant columns
    are handled by compute_variance_explained and compute_mse_per_column are not broken
    in the future.
    """
    nan = np.nan
    X = np.array(
        [[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.1, 1e-2, 1e-3, 0.0],
         [0.0, 2.0, 1.0, 1.0, 3.1, 1.0, 4.1, 0.0, 0.0, 1e-4],
         [0.0, 3.0, 1.0, 1.0, 7.1, 1.0, 6.0, 0.0, -1e-3, 0.0]]
    )
    # X_observed should not be used by the functions below since exclude_observed=False.
    # Setting it to nan should expose such a bug.
    X_observed = np.array(
        [[nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
         [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
         [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]]
    )
    X_completion = np.array(
        [[0.0, 1.0, 1.0, 1.0, 3.14, 1.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 2.0, 1.0, 1.0, 2.71, 1.0, 2.0, 0.0, 0.0, 0.0],
         [0.0, 3.0, 1.0, 1.0, 1.41, 1.0, 3.0, 0.0, 0.0, 0.0]]
    )
    variance_explained = compute_variance_explained(X, X_completion, give_each_column_equal_weight=True)
    mses = compute_mse_per_column(X, X_observed, X_completion, exclude_observed=False,
                                  give_each_column_equal_weight=True)
    print(mses)
    np.testing.assert_almost_equal(mses, np.array([0., 0., 0, 0., 1.93202707, 0., 2.34268186, 1.5, 1.0, 0.0]))
    X_standardized = standardize(X)
    variance_explained_2 = 1.0 - X.shape[0] * np.sum(mses) / np.sum(X_standardized ** 2)
    np.testing.assert_almost_equal(variance_explained, variance_explained_2)
    assert(~np.isnan(variance_explained))
    assert(~np.isnan(variance_explained_2))


def test_compute_variance_explained_and_mse_per_column_produce_nan():
    r"""
    If the model predictions are off, even only slightly, the variance explained and mse should be nan.
    """
    nan = np.nan
    X = np.array(
        [[1.0],
         [1.0],
         [1.0]]
    )
    X_observed = np.array(
        [[nan],
         [nan],
         [nan]]
    )
    X_completion = np.array(
        [[1.0 + 1e-2],
         [1.0],
         [1.0]]
    )
    variance_explained = compute_variance_explained(X, X_completion, give_each_column_equal_weight=True)
    mses = compute_mse_per_column(X, X_observed, X_completion, exclude_observed=False,
                                  give_each_column_equal_weight=True)
    print(mses)
    np.testing.assert_almost_equal(mses, np.array([nan]))
    X_standardized = standardize(X)
    variance_explained_2 = 1.0 - X.shape[0] * np.sum(mses) / np.sum(X_standardized ** 2)
    np.testing.assert_almost_equal(variance_explained, variance_explained_2)
    assert(np.isnan(variance_explained))
    assert(np.isnan(variance_explained_2))


def test_compute_pearson_r2s():
    nan = np.nan
    X = np.array(
        [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
         [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 2.0],
         [3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 3.0],
         [4.0, 4.0, 4.0, 4.0, 1.0, 1.0, 4.0],
         [5.0, 5.0, 5.0, 5.0, 1.0, 2.0, 5.0]])
    X_observed = np.array(
        [[1.0, nan, nan, nan, nan, nan, nan],
         [2.0, nan, 2.0, 2.0, nan, nan, nan],
         [3.0, nan, nan, nan, nan, nan, nan],
         [4.0, nan, 4.0, 4.0, 1.0, 1.0, nan],
         [5.0, nan, nan, nan, 1.0, 2.0, nan]])
    X_pred = np.array(
        [[1.0, 1.0, 5.0, 1.0, 9.0, 2.0, 0.0],
         [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.0],
         [3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 0.0],
         [4.0, 4.0, 4.0, 4.0, 1.0, 1.0, 0.0],
         [5.0, 5.0, 1.0, 9.0, 1.0, 2.0, 0.0]])
    pearson_r2s = compute_pearson_r2s(X, X_observed, X_pred, exclude_observed=False)
    pearson_r2s_expected = np.array([1.0, 1.0, -0.6, 0.9138115, 0.0, 0.6123724, 0.0])
    np.testing.assert_almost_equal(pearson_r2s, pearson_r2s_expected)

    pearson_r2s = compute_pearson_r2s(X, X_observed, X_pred, exclude_observed=True)
    pearson_r2s_expected = np.array([1.0, 1.0, -1.0, 0.9607689, 0.0, 0.0, 0.0])
    np.testing.assert_almost_equal(pearson_r2s, pearson_r2s_expected)


def test_compute_spearman_r2s():
    nan = np.nan
    X = np.array(
        [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
         [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 2.0],
         [3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 3.0],
         [4.0, 4.0, 4.0, 4.0, 1.0, 1.0, 4.0],
         [5.0, 5.0, 5.0, 5.0, 1.0, 2.0, 5.0]])
    X_observed = np.array(
        [[1.0, nan, nan, nan, nan, nan, nan],
         [2.0, nan, 2.0, 2.0, nan, nan, nan],
         [3.0, nan, nan, nan, nan, nan, nan],
         [4.0, nan, 4.0, 4.0, 1.0, 1.0, nan],
         [5.0, nan, nan, nan, 1.0, 2.0, nan]])
    X_pred = np.array(
        [[1.0, 1.0, 5.0, 1.0, 9.0, 2.0, 0.0],
         [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.0],
         [3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 0.0],
         [4.0, 4.0, 4.0, 4.0, 1.0, 1.0, 0.0],
         [5.0, 5.0, 1.0, 9.0, 1.0, 2.0, 0.0]])
    spearman_r2s = compute_spearman_r2s(X, X_observed, X_pred, exclude_observed=False)
    spearman_r2s_expected = np.array([1.0, 1.0, -0.6, 1.0, 0.0, 0.6123724, 0.0])
    np.testing.assert_almost_equal(spearman_r2s, spearman_r2s_expected)

    spearman_r2s = compute_spearman_r2s(X, X_observed, X_pred, exclude_observed=True)
    spearman_r2s_expected = np.array([1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_almost_equal(spearman_r2s, spearman_r2s_expected)


def test_compute_variance_explained_excluding_observed_entries():
    # TODO
    pass
