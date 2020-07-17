import numpy as np

from turbo_mc.models.cv_matrix_completion_model import compute_spearman_r2s, KFoldCVMatrixCompletionModel,\
    TrainValSplitCVMatrixCompletionModel
from turbo_mc.metrics import compute_spearman_r2s as compute_spearman_r2s_omniscient


def test_compute_spearman_r2s():
    nan = np.nan
    spearman_r2s = compute_spearman_r2s(
        X_true=np.array(
            [[1, nan, 3, 3, 3, 3],
             [nan, 5, 3, 6, 3, 6],
             [7, 8, nan, nan, nan, nan]]),
        X_pred=np.array(
            [[2, nan, 3, 3, 3, 3],
             [nan, 9, 3, 6, 6, 3],
             [6, 8, nan, nan, nan, nan]])
    )
    expected_spearman_r2s = np.array([1., -1., 1., 1., 0., 0.])
    np.testing.assert_almost_equal(spearman_r2s, expected_spearman_r2s)


def test_KFoldCVMatrixCompletionModel():
    r"""
    We create a random low rank matrix where two columns are random. Those columns should
    have the worse CV spearman R2s.
    """
    k_true = 1
    R = 1000
    C = 19
    np.random.seed(1)
    U = np.random.normal(size=(R, k_true))
    V = np.random.normal(size=(C, k_true))
    X_true = U @ V.T
    X_true[:, C // 2] = np.random.normal(size=(R))  # Make this column random.
    X_true[:, C // 3] = np.random.normal(size=(R))  # Make this column random.
    from turbo_mc.matrix_manipulation import get_list_of_random_matrix_indices
    np.random.seed(1)
    observed_indices = get_list_of_random_matrix_indices(
        X_true.shape[0], X_true.shape[1], sampling_density=0.5, randomize_rows_and_columns=True)
    from turbo_mc.matrix_manipulation import observe_entries_and_hide_rest
    X_observed = observe_entries_and_hide_rest(X_true, observed_indices)
    from turbo_mc.models.matrix_completion_fast_als import MatrixCompletionFastALS
    model = MatrixCompletionFastALS(n_factors=1, lam=0.0, n_epochs=100, verbose=False)
    cv_model = KFoldCVMatrixCompletionModel(model, n_folds=5, verbose=True, finally_refit_model=model)
    np.random.seed(1)
    cv_model.fit_matrix(X_observed)
    cv_spearman_r2s = cv_model.cv_spearman_r2s()
    print(cv_spearman_r2s)
    assert(set(np.argsort(cv_spearman_r2s)[:2]) == set([C // 2, C // 3]))
    assert(cv_spearman_r2s[C // 2] < 0.1)
    assert(cv_spearman_r2s[C // 3] < 0.1)
    X_completion = cv_model.predict_all()
    ground_truth_spearman_r2s = compute_spearman_r2s_omniscient(X_true, X_observed, X_completion, exclude_observed=True)
    assert(set(np.argsort(ground_truth_spearman_r2s)[:2]) == set([C // 2, C // 3]))
    assert(ground_truth_spearman_r2s[C // 2] < 0.1)
    assert(ground_truth_spearman_r2s[C // 3] < 0.1)
    np.testing.assert_almost_equal(cv_model.predict(0, 0), X_true[0, 0], decimal=2)


def test_TrainValSplitCVMatrixCompletionModel():
    r"""
    We create a random low rank matrix where two columns are random. Those columns should
    have the worse CV spearman R2s.
    """
    k_true = 1
    R = 1000
    C = 19
    np.random.seed(1)
    U = np.random.normal(size=(R, k_true))
    V = np.random.normal(size=(C, k_true))
    X_true = U @ V.T
    X_true[:, C // 2] = np.random.normal(size=(R))  # Make this column random.
    X_true[:, C // 3] = np.random.normal(size=(R))  # Make this column random.
    from turbo_mc.matrix_manipulation import get_list_of_random_matrix_indices
    np.random.seed(1)
    observed_indices = get_list_of_random_matrix_indices(
        X_true.shape[0], X_true.shape[1], sampling_density=0.5, randomize_rows_and_columns=True)
    from turbo_mc.matrix_manipulation import observe_entries_and_hide_rest
    X_observed = observe_entries_and_hide_rest(X_true, observed_indices)
    from turbo_mc.models.matrix_completion_fast_als import MatrixCompletionFastALS
    model = MatrixCompletionFastALS(n_factors=1, lam=0.0, n_epochs=100, verbose=False)
    cv_model = TrainValSplitCVMatrixCompletionModel(model, train_ratio=0.8, verbose=True, finally_refit_model=model)
    np.random.seed(1)
    cv_model.fit_matrix(X_observed)
    cv_spearman_r2s = cv_model.cv_spearman_r2s()
    print(cv_spearman_r2s)
    assert(set(np.argsort(cv_spearman_r2s)[:2]) == set([C // 2, C // 3]))
    assert(cv_spearman_r2s[C // 2] < 0.1)
    assert(cv_spearman_r2s[C // 3] < 0.1)
    X_completion = cv_model.predict_all()
    ground_truth_spearman_r2s = compute_spearman_r2s_omniscient(X_true, X_observed, X_completion, exclude_observed=True)
    assert(set(np.argsort(ground_truth_spearman_r2s)[:2]) == set([C // 2, C // 3]))
    assert(ground_truth_spearman_r2s[C // 2] < 0.1)
    assert(ground_truth_spearman_r2s[C // 3] < 0.1)
    np.testing.assert_almost_equal(cv_model.predict(0, 0), X_true[0, 0], decimal=2)
