import numpy as np

from turbo_mc.iterative_models.matrix_oracle import OracleWithAPrescribedMatrix,\
    WarmstartedOracleWithAPrescribedMatrix


def test_OracleWithAPrescribedMatrix():
    matrix_oracle = OracleWithAPrescribedMatrix(
        X=np.array(
            [[1, 2, 3],
             [4, 5, 6]]
        )
    )
    nan = np.nan

    X_observed = matrix_oracle.observed_matrix()
    X_observed_expected = np.array(
        [[nan, nan, nan],
         [nan, nan, nan]]
    )
    np.testing.assert_almost_equal(X_observed, X_observed_expected)

    matrix_indices = [(1, 0)]
    vals = matrix_oracle.observe_entries(matrix_indices)
    vals_expected = np.array([4])
    np.testing.assert_almost_equal(vals, vals_expected)
    X_observed = matrix_oracle.observed_matrix()
    X_observed_expected = np.array(
        [[nan, nan, nan],
         [4, nan, nan]]
    )
    np.testing.assert_almost_equal(X_observed, X_observed_expected)

    for repetition in range(2):  # Querying the same entries twice should work
        matrix_indices = [(0, 1), (1, 2)]
        vals = matrix_oracle.observe_entries(matrix_indices)
        vals_expected = np.array([2, 6])
        np.testing.assert_almost_equal(vals, vals_expected)
        X_observed = matrix_oracle.observed_matrix()
        X_observed_expected = np.array(
            [[nan, 2, nan],
             [4, nan, 6]]
        )
        np.testing.assert_almost_equal(X_observed, X_observed_expected)

    matrix_indices = [(1, 0), (1, 2), (0, 0), (1, 1), (0, 2), (0, 1)]
    vals = matrix_oracle.observe_entries(matrix_indices)
    vals_expected = np.array([4, 6, 1, 5, 3, 2])
    np.testing.assert_almost_equal(vals, vals_expected)
    X_observed = matrix_oracle.observed_matrix()
    X_observed_expected = np.array(
        [[1, 2, 3],
         [4, 5, 6]]
    )
    np.testing.assert_almost_equal(X_observed, X_observed_expected)


def test_warmstarting_matrix_oracle():
    matrix_oracle = WarmstartedOracleWithAPrescribedMatrix(
        X=np.array(
            [[1, 2, 3],
             [4, 5, 6]]
        ),
        warm_started_indices=[(1, 1), (0, 2)]
    )
    nan = np.nan

    X_observed = matrix_oracle.observed_matrix()
    X_observed_expected = np.array(
        [[nan, nan, 3],
         [nan, 5, nan]]
    )
    np.testing.assert_almost_equal(X_observed, X_observed_expected)

    matrix_indices = [(1, 0)]
    vals = matrix_oracle.observe_entries(matrix_indices)
    vals_expected = np.array([4])
    np.testing.assert_almost_equal(vals, vals_expected)
    X_observed = matrix_oracle.observed_matrix()
    X_observed_expected = np.array(
        [[nan, nan, 3],
         [4, 5, nan]]
    )
    np.testing.assert_almost_equal(X_observed, X_observed_expected)

    for repetition in range(2):  # Querying the same entries twice should work
        matrix_indices = [(0, 1), (1, 2)]
        vals = matrix_oracle.observe_entries(matrix_indices)
        vals_expected = np.array([2, 6])
        np.testing.assert_almost_equal(vals, vals_expected)
        X_observed = matrix_oracle.observed_matrix()
        X_observed_expected = np.array(
            [[nan, 2, 3],
             [4, 5, 6]]
        )
        np.testing.assert_almost_equal(X_observed, X_observed_expected)

    matrix_indices = [(1, 0), (1, 2), (0, 0), (1, 1), (0, 2), (0, 1)]
    vals = matrix_oracle.observe_entries(matrix_indices)
    vals_expected = np.array([4, 6, 1, 5, 3, 2])
    np.testing.assert_almost_equal(vals, vals_expected)
    X_observed = matrix_oracle.observed_matrix()
    X_observed_expected = np.array(
        [[1, 2, 3],
         [4, 5, 6]]
    )
    np.testing.assert_almost_equal(X_observed, X_observed_expected)