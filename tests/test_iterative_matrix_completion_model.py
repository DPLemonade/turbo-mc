import itertools
import numpy as np
import pytest

from turbo_mc.iterative_models.iterative_matrix_completion_model import IterativeMCMWithPrescribedPcts,\
    IterativeMCMWithGuaranteedSpearmanR2, _choose_entries_from_underperforming_columns,\
    _override_fully_observed_columns_sr2_to_1
from turbo_mc.iterative_models.matrix_oracle import OracleWithAPrescribedMatrix,\
    WarmstartedOracleWithAPrescribedMatrix
from turbo_mc.matrix_manipulation import get_list_of_random_matrix_indices
from turbo_mc.models.column_normalizer_model_wrapper import ColumnNormalizerModelWrapper
from turbo_mc.models.cv_matrix_completion_model import KFoldCVMatrixCompletionModel,\
    TrainValSplitCVMatrixCompletionModel
from turbo_mc.models.exclude_constant_columns_model_wrapper import ExcludeConstantColumnsModelWrapper
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

    # Check that the random columns are among the most observed.
    X_observed = iterative_model.observed_matrix()
    observations_per_column = (~np.isnan(X_observed)).sum(axis=0)
    print(f"observations_per_column = {observations_per_column}")
    for col in [C // 2, C // 3, C // 4, C // 5]:
        for other_col in range(C):
            assert(observations_per_column[col] >= observations_per_column[other_col])

    # Check that random columns have best SR2
    iterative_model.predict_all()
    cv_spearman_r2s = iterative_model.cv_spearman_r2s()
    assert(min(cv_spearman_r2s) > requested_cv_spearman_r2)
    for col in [C // 2, C // 3, C // 4, C // 5]:
        for other_col in range(C):
            assert(cv_spearman_r2s[col] >= cv_spearman_r2s[other_col])


@pytest.mark.parametrize(
    "finally_refit_model",
    [None,
     MatrixCompletionFastALS(
         n_factors=1,
         lam=0.0,
         n_epochs=100,
         verbose=False)]
)
def test_IterativeMCMWithGuaranteedSpearmanR2__small_warmstarting_all_matrix(finally_refit_model):
    r"""
    We create a random low rank matrix where four columns are random.
    We start from a warmstarted oracle where all entries were observed,
    so we should be able to obtain complete reconstruction in 1 iteration.
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
    matrix_oracle = WarmstartedOracleWithAPrescribedMatrix(
        X_true,
        list(itertools.product(range(R), range(C))))
    requested_cv_spearman_r2 = 0.99
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
        verbose=True,
        max_iterations=1
    )
    iterative_model.fit(matrix_oracle)
    X_observed = iterative_model.observed_matrix()
    observations_per_column = (~np.isnan(X_observed)).sum(axis=0)
    assert(np.all(observations_per_column == R))
    iterative_model.predict_all()
    cv_spearman_r2s = iterative_model.cv_spearman_r2s()
    assert(min(cv_spearman_r2s) > requested_cv_spearman_r2)


np.random.seed(1)


@pytest.mark.parametrize(
    "finally_refit_model, cv_model, warmstarted_entries",
    itertools.product(
        [None,
         MatrixCompletionFastALS(
             n_factors=1,
             lam=0.0,
             n_epochs=100,
             verbose=False)],
        [KFoldCVMatrixCompletionModel(
            model=MatrixCompletionFastALS(
                n_factors=1,
                lam=0.0,
                n_epochs=100,
                verbose=False),
            n_folds=5,
            finally_refit_model=None),
         TrainValSplitCVMatrixCompletionModel(
            model=MatrixCompletionFastALS(
                n_factors=1,
                lam=0.0,
                n_epochs=100,
                verbose=False),
            train_ratio=0.8,
            finally_refit_model=None
        )],
        [[(16, 16)],  # Just one entry
         [(0, 0),
          (1, 1), (2, 1),
          (3, 2), (4, 2), (5, 2),
          (6, 3), (7, 3), (8, 3), (9, 3),
          (10, 4), (11, 4), (12, 4), (13, 4), (14, 4)],  # Few with != num obs per column
         get_list_of_random_matrix_indices(37, 17, 0.9),  # Many random
         get_list_of_random_matrix_indices(37, 17, 0.75),  # Many random
         get_list_of_random_matrix_indices(37, 17, 0.5),  # Half random
         get_list_of_random_matrix_indices(37, 17, 0.25),  # Quarter random
         get_list_of_random_matrix_indices(37, 17, 0.1),  # Quarter random
         ]
    )
)
def test_IterativeMCMWithGuaranteedSpearmanR2__small_warmstarting(
    finally_refit_model, cv_model, warmstarted_entries
):
    r"""
    We create a random low rank matrix where four columns are random.
    We start from a warmstarted oracle with diverse characteristics
    (few entries observed / distributed differently / many observed),
    Checks should still pass (e.g. most queried columns are the random ones),
    and code shouldn't blow up!
    This test is pretty exhaustive as it also stresses several underlying CV models
    (arguably, the ability of CV models to be fit on weird observation patters should
    NOT belong here, but we're just doing this as a 'plus'...)
    Same goes for finally refitting the model.
    If this test becomes too slow, either finally_refit_model and/or cv_model can
    be chopped out of the cross product.
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
    matrix_oracle = WarmstartedOracleWithAPrescribedMatrix(X_true, warmstarted_entries)
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

    # Check that the random columns are among the most observed.
    X_observed = iterative_model.observed_matrix()
    observations_per_column = (~np.isnan(X_observed)).sum(axis=0)
    print(f"observations_per_column = {observations_per_column}")
    for col in [C // 2, C // 3, C // 4, C // 5]:
        for other_col in range(C):
            assert(observations_per_column[col] >= observations_per_column[other_col])

    # Check that random columns have best SR2
    iterative_model.predict_all()
    cv_spearman_r2s = iterative_model.cv_spearman_r2s()
    assert(min(cv_spearman_r2s) > requested_cv_spearman_r2)
    for col in [C // 2, C // 3, C // 4, C // 5]:
        for other_col in range(C):
            assert(cv_spearman_r2s[col] >= cv_spearman_r2s[other_col])


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


def test_IterativeMCMWithGuaranteedSpearmanR2__large_warmstarting():
    r"""
    We create a random low rank matrix where four columns are random. Those four columns should
    all be queried fully by the model! The oracle is also heavily warmstarstarted, which
    is why this is a fast test!: If the sampling_density used for the matrix_oracle
    is reduced to e.g. 0.01 this test takes longer to run (as expected).
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
    matrix_oracle = WarmstartedOracleWithAPrescribedMatrix(
        X_true, get_list_of_random_matrix_indices(R, C, sampling_density=0.75))
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
        finally_refit_model=MatrixCompletionFastALS(
            n_factors=1,
            lam=0.0,
            n_epochs=100,
            verbose=False),
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


@pytest.mark.parametrize(
    "cv_model",
    [lambda state:
        TrainValSplitCVMatrixCompletionModel(
            ExcludeConstantColumnsModelWrapper(
                ColumnNormalizerModelWrapper(
                    MatrixCompletionFastALS(
                        n_factors=int(min(state.R, state.C) * state.sampled_density * 0.5),
                        lam=10.0,
                        n_epochs=100,
                        verbose=False))),
            train_ratio=0.8,
            verbose=True,
            finally_refit_model=None),
     lambda state:
        KFoldCVMatrixCompletionModel(
            ExcludeConstantColumnsModelWrapper(
                ColumnNormalizerModelWrapper(
                    MatrixCompletionFastALS(
                        n_factors=int(min(state.R, state.C) * state.sampled_density * 0.5),
                        lam=10.0,
                        n_epochs=100,
                        verbose=False))),
            n_folds=5,
            verbose=True,
            finally_refit_model=None),
     lambda state:
        TrainValSplitCVMatrixCompletionModel(
            ColumnNormalizerModelWrapper(
                MatrixCompletionFastALS(
                    n_factors=int(min(state.R, state.C) * state.sampled_density * 0.5),
                    lam=10.0,
                    n_epochs=100,
                    verbose=False)),
            train_ratio=0.8,
            verbose=True,
            finally_refit_model=None),
     lambda state:
        KFoldCVMatrixCompletionModel(
            ColumnNormalizerModelWrapper(
                MatrixCompletionFastALS(
                    n_factors=int(min(state.R, state.C) * state.sampled_density * 0.5),
                    lam=10.0,
                    n_epochs=100,
                    verbose=False)),
            n_folds=5,
            verbose=True,
            finally_refit_model=None),
     lambda state:
        TrainValSplitCVMatrixCompletionModel(
            ExcludeConstantColumnsModelWrapper(
                MatrixCompletionFastALS(
                    n_factors=int(min(state.R, state.C) * state.sampled_density * 0.5),
                    lam=10.0,
                    n_epochs=100,
                    verbose=False)),
            train_ratio=0.8,
            verbose=True,
            finally_refit_model=None),
     lambda state:
        KFoldCVMatrixCompletionModel(
            ExcludeConstantColumnsModelWrapper(
                MatrixCompletionFastALS(
                    n_factors=int(min(state.R, state.C) * state.sampled_density * 0.5),
                    lam=10.0,
                    n_epochs=100,
                    verbose=False)),
            n_folds=5,
            verbose=True,
            finally_refit_model=None),
     lambda state:
        TrainValSplitCVMatrixCompletionModel(
            ColumnNormalizerModelWrapper(
                ExcludeConstantColumnsModelWrapper(
                    MatrixCompletionFastALS(
                        n_factors=int(min(state.R, state.C) * state.sampled_density * 0.5),
                        lam=10.0,
                        n_epochs=100,
                        verbose=False))),
            train_ratio=0.8,
            verbose=True,
            finally_refit_model=None),
     lambda state:
        KFoldCVMatrixCompletionModel(
            ColumnNormalizerModelWrapper(
                ExcludeConstantColumnsModelWrapper(
                    MatrixCompletionFastALS(
                        n_factors=int(min(state.R, state.C) * state.sampled_density * 0.5),
                        lam=10.0,
                        n_epochs=100,
                        verbose=False))),
            n_folds=5,
            verbose=True,
            finally_refit_model=None)]
)
def test_integration_with_normalization_and_exclusion_wrappers(cv_model):
    r"""
    When fitting on a 2 x C matrix, things blew up for me in Compass.
    It would be good to be able to run successfully on a 2 x C matrix for Compass tests!
    This test case debugs the issue and stands as a regression test.
    """
    X = np.array([[0., 1., 2., 3.], [4., 3., 2., 1.]])
    compass_matrix_oracle = OracleWithAPrescribedMatrix(X)
    requested_cv_spearman_r2 = 0.7
    increments = 0.5
    min_pct_meet_sr2_requirement = 0.99
    model =\
        IterativeMCMWithGuaranteedSpearmanR2(
            cv_model=cv_model,
            requested_cv_spearman_r2=requested_cv_spearman_r2,
            sampling_density=increments,
            finally_refit_model=lambda state:
                ExcludeConstantColumnsModelWrapper(
                    ColumnNormalizerModelWrapper(
                        MatrixCompletionFastALS(
                            n_factors=int(min(state.R, state.C) * state.sampled_density * 0.5),
                            lam=10.0,
                            n_epochs=300,
                            verbose=False))),
            min_pct_meet_sr2_requirement=min_pct_meet_sr2_requirement,
            verbose=True,
            plot_progress=False,
        )

    # Fit iterative model. This is the core procedure of Turbo Compass (here is where all the magic happens)
    np.random.seed(1)
    model.fit(compass_matrix_oracle)
