import itertools
import numpy as np
import pytest

from turbo_mc.iterative_models.iterative_matrix_completion_model import IterativeMCMWithPrescribedPcts,\
    IterativeMCMWithGuaranteedSpearmanR2, _choose_entries_from_underperforming_columns,\
    _override_fully_observed_columns_sr2_to_1
from turbo_mc.iterative_models.matrix_oracle import OracleWithAPrescribedMatrix
from turbo_mc.models.cv_matrix_completion_model import KFoldCVMatrixCompletionModel
from turbo_mc.models.matrix_completion_fast_als import MatrixCompletionFastALS
np.set_printoptions(precision=2, suppress=True)


def test_IterativeMCMWithPrescribedPcts():
    r"""
    We create a random low rank matrix where four columns are random. Those columns should
    all be queried more than the other ones.
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
    X_true[:, C // 4] = np.random.normal(size=(R))  # Make this column random.
    X_true[:, C // 5] = np.random.normal(size=(R))  # Make this column random.
    matrix_oracle = OracleWithAPrescribedMatrix(X_true)
    iterative_model = IterativeMCMWithPrescribedPcts(
        [0.5, 0.1],
        [KFoldCVMatrixCompletionModel(MatrixCompletionFastALS(n_factors=1, lam=0.0, n_epochs=100, verbose=False),
                                      n_folds=5, verbose=True),
         KFoldCVMatrixCompletionModel(
            MatrixCompletionFastALS(n_factors=1, lam=0.0, n_epochs=100, verbose=False),
            n_folds=5, verbose=True,
            finally_refit_model=MatrixCompletionFastALS(n_factors=1, lam=0.0, n_epochs=300, verbose=False))],
        verbose=True)
    iterative_model.fit(matrix_oracle)
    X_observed = iterative_model.observed_matrix()
    observations_per_column = (~np.isnan(X_observed)).sum(axis=0)
    top_queried_columns = np.argsort(observations_per_column)[-(C // 2):]
    assert(all([c in top_queried_columns for c in [C // 2, C // 3, C // 4, C // 5]]))
    iterative_model.predict_all()
    cv_spearman_r2s = iterative_model.cv_spearman_r2s()
    poorest_spearman_r2s = np.argsort(cv_spearman_r2s)[:(C // 2)]
    assert(all([c in poorest_spearman_r2s for c in [C // 2, C // 3, C // 4, C // 5]]))


def test_IterativeMCMWithPrescribedPcts_raises():
    r"""
    The sampling_densities list and cv_models list should have the same length
    or else we should raise.
    """
    with pytest.raises(ValueError):
        IterativeMCMWithPrescribedPcts([0.2, 0.2], [None])


def test__override_fully_observed_columns_sr2_to_1():
    nan = np.nan
    X_observed = np.array(
        [[nan, 2., 3., 1],
         [3., 4., 1., nan],
         [nan, 6., 9., nan]]
    )
    curr_cv_spearman_r2s = np.array([0.1, 0.2, 0.3, 0.4])
    _override_fully_observed_columns_sr2_to_1(curr_cv_spearman_r2s, X_observed)
    np.testing.assert_almost_equal(curr_cv_spearman_r2s, [0.1, 1.0, 1.0, 0.4])


def test__choose_entries_from_underperforming_columns__small():
    nan = np.nan
    X_observed = np.array(
        [[1.0, nan, 3.0, nan],
         [nan, 6.0, 7.0, nan],
         [9.0, nan, 1.0, nan]]
    )
    cv_spearman_r2s = np.array([0.8, 0.2, 0.9, 0.1])
    for requested_cv_spearman_r2 in np.arange(0.1, 1.01, 0.05):
        for sampling_density in np.arange(0.1, 1.01, 0.05):
            np.random.seed(1)
            res =\
                _choose_entries_from_underperforming_columns(
                    X_observed,
                    sampling_density,
                    cv_spearman_r2s,
                    requested_cv_spearman_r2)
            # Check that matrix indices are indeed unobserved, and that the number of observed entries is correct.
            for (r, c) in res:
                assert(np.isnan(X_observed[r, c]))
            R, C = X_observed.shape
            assert(len(res) == min(int(sampling_density * R * C), np.isnan(X_observed).sum().sum()))
            # Some specific checks by hand
            if len(res) == 5 and abs(requested_cv_spearman_r2 - 0.7) < 1e-7:
                assert(set(res) == set([(0, 3), (1, 3), (2, 3), (0, 1), (2, 1)]))
            if len(res) == 1 and abs(requested_cv_spearman_r2 - 0.7) < 1e-7:
                assert(set(res) in [{(0, 3)}, {(1, 3)}, {(2, 3)}])
            if len(res) == 3 and abs(requested_cv_spearman_r2 - 0.15) < 1e-7:
                # Only the poorest column should be queried!
                assert(set(res) == {(0, 3), (1, 3), (2, 3)})
            if len(res) == 2 and abs(requested_cv_spearman_r2 - 0.15) < 1e-7:
                # Only the poorest column should be queried!
                assert(set(res) in [{(0, 3), (1, 3)}, {(0, 3), (2, 3)}, {(1, 3), (2, 3)}])
            if len(res) == 1 and abs(requested_cv_spearman_r2 - 0.15) < 1e-7:
                # Only the poorest column should be queried!
                assert(set(res) in [{(0, 3)}, {(1, 3)}, {(2, 3)}])


@pytest.mark.parametrize(
    "R, C",
    [(7, 37),
     (37, 7),
     (1, 37),
     (37, 1),
     (1, 1)]
    )
def test__choose_entries_from_underperforming_columns__medium(R, C):
    r"""
    Creates a matrix with column c observed to depth c / (C - 1): i.e. columns
    to the left are less observed; column 0 is totally unobserved;
    column C-1 is totally observed).

    Loops over a grid of requested_cv_spearman_r2 X sampling_density and makes
    sure that the returned entries are indeed unobserved and in the quantity requested.
    """
    nan = np.nan
    np.random.seed(1)
    X_observed = np.random.normal(size=(R, C))
    for c in range(C):
        for r in range(R):
            if np.random.uniform(size=(1))[0] < c / (C - 1 + 1e-8):
                X_observed[r, c] = nan
    print(X_observed)
    key = 0
    for requested_cv_spearman_r2 in np.arange(0.05, 1.01, 0.05):
        for sampling_density in np.arange(0.1, 1.01, 0.05):
            key += 1
            np.random.seed(key)
            cv_spearman_r2s = np.random.uniform(size=(C))
            # Force a col to spearman R2 zero. This scheme guarantees all columns
            # appear as poor columns at least one time.
            cv_spearman_r2s[key % C] = 0.0
            np.random.seed(1)
            res =\
                _choose_entries_from_underperforming_columns(
                    X_observed,
                    sampling_density,
                    cv_spearman_r2s,
                    requested_cv_spearman_r2)
            # Check that matrix indices are indeed unobserved, and that the number of observed entries is correct.
            for (r, c) in res:
                assert(np.isnan(X_observed[r, c]))
            assert(len(res) == min(int(sampling_density * R * C), np.isnan(X_observed).sum().sum()))


@pytest.mark.parametrize(
    "finally_refit_model",
    [None,
     MatrixCompletionFastALS(
        n_factors=1,
        lam=0.0,
        n_epochs=100,
        verbose=False)]
    )
def test_IterativeMCMWithGuaranteedSpearmanR2__small(finally_refit_model):
    r"""
    We create a random low rank matrix where four columns are random. Those four columns should
    all be queried fully by the model!
    """
    k_true = 1
    R = 37
    C = 17
    np.random.seed(1)
    U = np.random.normal(size=(R, k_true))
    V = np.random.normal(size=(C, k_true))
    X_true = U @ V.T
    X_true[:, C // 2] = np.random.normal(size=(R))  # Make this column random.
    X_true[:, C // 3] = np.random.normal(size=(R))  # Make this column random.
    X_true[:, C // 4] = np.random.normal(size=(R))  # Make this column random.
    X_true[:, C // 5] = np.random.normal(size=(R))  # Make this column random.
    matrix_oracle = OracleWithAPrescribedMatrix(X_true)
    requested_cv_spearman_r2 = 0.7
    iterative_model = IterativeMCMWithGuaranteedSpearmanR2(
        cv_model=KFoldCVMatrixCompletionModel(
            model=MatrixCompletionFastALS(
                n_factors=1,
                lam=0.0,
                n_epochs=100,
                verbose=False),
            n_folds=5,
            finally_refit_model=None),
        requested_cv_spearman_r2=requested_cv_spearman_r2,
        sampling_density=0.05,
        finally_refit_model=finally_refit_model,
        verbose=True
    )
    iterative_model.fit(matrix_oracle)
    X_observed = iterative_model.observed_matrix()
    observations_per_column = (~np.isnan(X_observed)).sum(axis=0)
    print(f"observations_per_column = {observations_per_column}")
    top_queried_columns = np.argsort(observations_per_column)[-4:]
    assert(set(top_queried_columns) == set([C // 2, C // 3, C // 4, C // 5]))
    iterative_model.predict_all()
    cv_spearman_r2s = iterative_model.cv_spearman_r2s()
    assert(min(cv_spearman_r2s) > requested_cv_spearman_r2)
    best_spearman_r2s = np.argsort(cv_spearman_r2s)[-4:]
    assert(set(best_spearman_r2s) == set([C // 2, C // 3, C // 4, C // 5]))


@pytest.mark.slow
@pytest.mark.parametrize(
    "finally_refit_model",
    [None,
     MatrixCompletionFastALS(
        n_factors=1,
        lam=0.0,
        n_epochs=100,
        verbose=False)]
    )
def test_IterativeMCMWithGuaranteedSpearmanR2__large(finally_refit_model):
    r"""
    We create a random low rank matrix where four columns are random. Those four columns should
    all be queried fully by the model!
    """
    k_true = 1
    R = 100
    C = 37
    np.random.seed(1)
    U = np.random.normal(size=(R, k_true))
    V = np.random.normal(size=(C, k_true))
    X_true = U @ V.T
    X_true[:, C // 2] = np.random.normal(size=(R))  # Make this column random.
    X_true[:, C // 3] = np.random.normal(size=(R))  # Make this column random.
    X_true[:, C // 4] = np.random.normal(size=(R))  # Make this column random.
    X_true[:, C // 5] = np.random.normal(size=(R))  # Make this column random.
    matrix_oracle = OracleWithAPrescribedMatrix(X_true)
    requested_cv_spearman_r2 = 0.7
    iterative_model = IterativeMCMWithGuaranteedSpearmanR2(
        cv_model=KFoldCVMatrixCompletionModel(
            model=MatrixCompletionFastALS(
                n_factors=1,
                lam=0.0,
                n_epochs=100,
                verbose=False),
            n_folds=5,
            finally_refit_model=None),
        requested_cv_spearman_r2=requested_cv_spearman_r2,
        sampling_density=0.01,
        finally_refit_model=finally_refit_model,
        verbose=True
    )
    iterative_model.fit(matrix_oracle)
    X_observed = iterative_model.observed_matrix()
    observations_per_column = (~np.isnan(X_observed)).sum(axis=0)
    print(f"observations_per_column = {observations_per_column}")
    top_queried_columns = np.argsort(observations_per_column)[-4:]
    assert(set(top_queried_columns) == set([C // 2, C // 3, C // 4, C // 5]))
    iterative_model.predict_all()
    iterative_model.impute_all()
    cv_spearman_r2s = iterative_model.cv_spearman_r2s()
    assert(min(cv_spearman_r2s) > requested_cv_spearman_r2)
    best_spearman_r2s = np.argsort(cv_spearman_r2s)[-4:]
    assert(set(best_spearman_r2s) == set([C // 2, C // 3, C // 4, C // 5]))


@pytest.mark.parametrize(
    "cv_model,finally_refit_model",
    itertools.product(
        [KFoldCVMatrixCompletionModel(
            model=MatrixCompletionFastALS(
                n_factors=1,
                lam=0.0,
                n_epochs=1,
                verbose=False)),
         lambda state: KFoldCVMatrixCompletionModel(
            model=MatrixCompletionFastALS(
                n_factors=max(1, int(min(state.R, state.C) * state.sampled_density * 0.5)),
                lam=0.0,
                n_epochs=1,
                verbose=False))],  # cv_model
        [None,
         MatrixCompletionFastALS(
            n_factors=1,
            lam=0.0,
            n_epochs=1,
            verbose=False),
         lambda state: MatrixCompletionFastALS(
            n_factors=max(1, int(min(state.R, state.C) * state.sampled_density * 0.5)),
            lam=0.0,
            n_epochs=1,
            verbose=False)]  # finally_refit_model
        )
    )
def test_IterativeMCMWithGuaranteedSpearmanR2__callables(cv_model, finally_refit_model):
    r"""
    Test that supplying callables to IterativeMCMWithGuaranteedSpearmanR2 works.
    It is similar to other tests but we ignore functonality, just make sure the calls work
    for each user case (i.e. this is more of a smoke test).
    """
    k_true = 1
    R = 37
    C = 17
    np.random.seed(1)
    U = np.random.normal(size=(R, k_true))
    V = np.random.normal(size=(C, k_true))
    X_true = U @ V.T
    X_true[:, C // 2] = np.random.normal(size=(R))  # Make this column random.
    X_true[:, C // 3] = np.random.normal(size=(R))  # Make this column random.
    X_true[:, C // 4] = np.random.normal(size=(R))  # Make this column random.
    X_true[:, C // 5] = np.random.normal(size=(R))  # Make this column random.
    matrix_oracle = OracleWithAPrescribedMatrix(X_true)
    requested_cv_spearman_r2 = 0.0
    iterative_model = IterativeMCMWithGuaranteedSpearmanR2(
        cv_model=cv_model,
        requested_cv_spearman_r2=requested_cv_spearman_r2,
        sampling_density=0.05,
        finally_refit_model=finally_refit_model,
        verbose=True
    )
    iterative_model.fit(matrix_oracle)
    iterative_model.observed_matrix()
    iterative_model.predict_all()
    iterative_model.cv_spearman_r2s()
