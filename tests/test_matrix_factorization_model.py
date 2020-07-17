import numpy as np
import pandas as pd
import pytest

from turbo_mc.models.matrix_factorization_model import MatrixFactorizationModel
from turbo_mc.matrix_manipulation import observe_entries, get_list_of_random_matrix_indices,\
    get_observations_from_partially_observed_matrix, matrix_from_observations,\
    complete_matrix
from turbo_mc.metrics import compute_r2s


def test_MatrixFactorizationModel_1():
    X = np.array(
        [[1, np.nan],
         [1, 1]]
    )
    observations = get_observations_from_partially_observed_matrix(X)
    model = MatrixFactorizationModel(n_factors=1, random_state=42)
    model.fit(observations, nrows=X.shape[0], ncols=X.shape[1])
    prediction = model.predict(0, 1)
    assert(abs(prediction - 1) < 1e-1)


def test_MatrixFactorizationModel_2():
    X = np.array(
        [[5, 6, 8],
         [11, 14, 18],
         [17, np.nan, 28]]
    )
    observations = get_observations_from_partially_observed_matrix(X)
    model = MatrixFactorizationModel(n_factors=2, random_state=42)
    model.fit(observations, nrows=X.shape[0], ncols=X.shape[1])
    prediction = model.predict(2, 1)
    assert(abs(prediction - 22) < 1e-1)


def test_MatrixFactorizationModel_3():
    np.random.seed(1)
    nrows = 20
    ncols = 19
    true_k = 4
    model_k = 4
    sampling_density = 0.5
    U = np.random.normal(size=(nrows, true_k))
    V = np.random.normal(size=(true_k, ncols))
    X = U @ V
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
    assert(mse < 1e-8)


def test_MatrixFactorizationModel_3_biased_predict_all():
    nan = np.nan
    X = np.array(
        [[3, 5],
         [4, 6],
         [5, 7]]
    )
    R, C = X.shape
    biased = True
    model_k = 1
    sampling_density = 1.0
    nrows, ncols = X.shape
    mf_model = MatrixFactorizationModel(n_factors=model_k, reg_all=0.0, biased=biased)
    random_matrix_indices = get_list_of_random_matrix_indices(nrows, ncols, sampling_density)
    observed_entries = observe_entries(X, random_matrix_indices)
    mf_model.fit(observed_entries, nrows=X.shape[0], ncols=X.shape[1])
    X_predict_all = mf_model.predict_all()
    X_completed_with_complete_matrix = complete_matrix(np.array([[nan, nan], [nan, nan], [nan, nan]]), mf_model)
    np.testing.assert_almost_equal(X_completed_with_complete_matrix, X, decimal=2)
    np.testing.assert_almost_equal(X_predict_all, X, decimal=2)
    X_completed_with_predict = np.zeros(shape=(R, C))
    for r in range(R):
        for c in range(C):
            X_completed_with_predict[r, c] = mf_model.predict(r, c)
    np.testing.assert_almost_equal(X_predict_all, X_completed_with_predict)


def test_MatrixFactorizationModel_3_unbiased_predict_all():
    nan = np.nan
    X = np.array(
        [[3, 5],
         [4, 6],
         [5, 7]]
    )
    R, C = X.shape
    biased = False
    model_k = 1
    sampling_density = 1.0
    nrows, ncols = X.shape
    mf_model = MatrixFactorizationModel(n_factors=model_k, reg_all=0.0, biased=biased)
    random_matrix_indices = get_list_of_random_matrix_indices(nrows, ncols, sampling_density)
    observed_entries = observe_entries(X, random_matrix_indices)
    mf_model.fit(observed_entries, nrows=X.shape[0], ncols=X.shape[1])
    X_predict_all = mf_model.predict_all()
    X_completed_with_complete_matrix = complete_matrix(np.array([[nan, nan], [nan, nan], [nan, nan]]), mf_model)
    X_completed_with_predict = np.zeros(shape=(R, C))
    for r in range(R):
        for c in range(C):
            X_completed_with_predict[r, c] = mf_model.predict(r, c)
    with pytest.raises(AssertionError):
        np.testing.assert_almost_equal(X_completed_with_complete_matrix, X, decimal=3)
        np.testing.assert_almost_equal(X_predict_all, X_completed_with_complete_matrix, decimal=3)
        np.testing.assert_almost_equal(X_predict_all, X_completed_with_predict)
    np.testing.assert_almost_equal(X_predict_all, X_completed_with_predict)
    np.testing.assert_almost_equal(X_predict_all, X_completed_with_complete_matrix)


def test_MatrixFactorizationModel_3_variant_predict_all():
    r"""
    Tests that predict_all works, which is much faster than complete_matrix
    """
    np.random.seed(1)
    nrows = 20
    ncols = 19
    true_k = 4
    model_k = 4
    sampling_density = 0.5
    U = np.random.normal(size=(nrows, true_k))
    V = np.random.normal(size=(true_k, ncols))
    X = U @ V
    mf_model = MatrixFactorizationModel(n_factors=model_k, reg_all=0.0)
    random_matrix_indices = get_list_of_random_matrix_indices(nrows, ncols, sampling_density)
    observed_entries = observe_entries(X, random_matrix_indices)
    mf_model.fit(observed_entries, nrows=X.shape[0], ncols=X.shape[1])
    X_observed = matrix_from_observations(observed_entries, nrows=X.shape[0], ncols=X.shape[1])
    unobserved_indices = np.where(np.isnan(X_observed))
    X_completion = X_observed.copy()
    X_completion[unobserved_indices] = mf_model.predict_all()[unobserved_indices]
    error = np.array(X_completion[unobserved_indices] - X[unobserved_indices])
    mse = np.mean((error * error))
    print(f"MSE = {mse}")
    assert(mse < 1e-8)
    error = mf_model.predict_all() - X
    mse = np.mean((error * error))
    print(f"MSE = {mse}")
    assert(mse < 1e-8)


@pytest.mark.slow
def test_MatrixFactorizationModel_4():
    np.random.seed(1)
    nrows = 290
    ncols = 6000
    true_k = 50
    model_k = 50
    sampling_density = 0.5
    U = np.random.normal(size=(nrows, true_k))
    V = np.random.normal(size=(true_k, ncols))
    X = U @ V
    mf_model = MatrixFactorizationModel(n_factors=model_k, reg_all=0.0, n_epochs=10, lr_all=0.003, verbose=True)
    random_matrix_indices = get_list_of_random_matrix_indices(nrows, ncols, sampling_density, verbose=True)
    observed_entries = observe_entries(X, random_matrix_indices, verbose=True)
    mf_model.fit(observed_entries, nrows=X.shape[0], ncols=X.shape[1])
    X_observed = matrix_from_observations(observed_entries, nrows=X.shape[0], ncols=X.shape[1])
    X_completion = mf_model.impute_all(X_observed)
    unobserved_indices = np.where(np.isnan(X_observed))
    error = np.array(X_completion[unobserved_indices] - X[unobserved_indices])
    mse = np.mean((error * error))
    print(f"MSE = {mse}")
    assert(mse < 1e-2)


@pytest.mark.slow
def test_MatrixFactorizationModel_5():
    np.random.seed(1)
    nrows = 290
    ncols = 6000
    true_k = 1
    model_k = 1
    sampling_density = 0.1
    U = np.random.normal(size=(nrows, true_k))
    V = np.random.normal(size=(true_k, ncols))
    X = U @ V
    mf_model = MatrixFactorizationModel(n_factors=model_k, reg_all=0.0, n_epochs=100, lr_all=0.003, verbose=True)
    random_matrix_indices = get_list_of_random_matrix_indices(nrows, ncols, sampling_density, verbose=True)
    observed_entries = observe_entries(X, random_matrix_indices, verbose=True)
    mf_model.fit(observed_entries, nrows=X.shape[0], ncols=X.shape[1])
    X_observed = matrix_from_observations(observed_entries, nrows=X.shape[0], ncols=X.shape[1])
    X_completion = mf_model.impute_all(X_observed)
    unobserved_indices = np.where(np.isnan(X_observed))
    error = np.array(X_completion[unobserved_indices] - X[unobserved_indices])
    mse = np.mean((error * error))
    print(f"MSE = {mse}")
    assert(mse < 1e-8)


def test_with_fake_reaction_data():
    # cell_X_metadata = pd.read_table("tests/data/fake/fake_metadata.csv", sep=",")
    reaction_X_cell_df = pd.read_csv("tests/data/fake/fake_reactions.csv", delimiter=',')
    reaction_X_cell_df.rename(columns={'Unnamed: 0': 'reaction'}, inplace=True)
    reaction_X_cell_df.set_index('reaction', inplace=True)
    # reaction_names = pd.DataFrame({'reaction_name': reaction_X_cell_df.index})
    cell_X_reaction_df = reaction_X_cell_df.transpose()
    del reaction_X_cell_df
    X = np.array(cell_X_reaction_df)

    np.random.seed(1)
    random_matrix_indices = get_list_of_random_matrix_indices(X.shape[0], X.shape[1], 0.5)
    observations = observe_entries(X, random_matrix_indices)
    X_observed = matrix_from_observations(observations, nrows=X.shape[0], ncols=X.shape[1])
    mf_model = MatrixFactorizationModel(
        n_factors=1,
        reg_all=0.0,
        n_epochs=100,
        lr_all=0.01,
        verbose=False)
    mf_model.fit(observations, nrows=X.shape[0], ncols=X.shape[1])
    X_completion = mf_model.impute_all(X_observed)
    print("X = ", X)
    print("X_observed = ", X_observed)
    print("X_completion = ", X_completion)
    r2s = compute_r2s(X, X_observed, X_completion)
    expected_rs2 = np.array(
        [0.9652624, 0.9464919, 0.9636387, 0.9305042,
         0.9593281, 0.9724987, 0.0318134, 0.0,
         -1.0, 1.0, 1.0, 1.0])
    np.testing.assert_almost_equal(r2s, expected_rs2)


@pytest.mark.parametrize("biased", [False, True])
def test_MatrixFactorizationModel_3_predict_all(biased):
    r"""
    Tests that predict_all works, which is much faster than complete_matrix
    """
    R = 20
    C = 19
    X = np.random.normal(size=(R, C))
    for r in range(R):
        for c in range(C):
            if (r + c) % 3 != 0:
                X[r, c] = np.nan
    observations = get_observations_from_partially_observed_matrix(X)
    model = MatrixFactorizationModel(n_factors=2, n_epochs=10, biased=biased)
    model.fit(observations, nrows=X.shape[0], ncols=X.shape[1])
    X_completed_with_predict = X.copy()
    for r in range(R):
        for c in range(C):
            if np.isnan(X[r, c]):
                X_completed_with_predict[r, c] = model.predict(r, c)
    X_completed_with_predict_all = X.copy()
    unobserved_indices = np.where(np.isnan(X))
    X_completed_with_predict_all[unobserved_indices] = model.predict_all()[unobserved_indices]
    np.testing.assert_almost_equal(X_completed_with_predict_all, X_completed_with_predict)
    X_completed_with_complete_matrix = complete_matrix(X, model)
    np.testing.assert_almost_equal(X_completed_with_predict_all, X_completed_with_complete_matrix)
    X_completed_with_impute_all = model.impute_all(X)
    np.testing.assert_almost_equal(X_completed_with_predict_all, X_completed_with_impute_all)


@pytest.mark.parametrize("biased", [False, True])
def test_with_fake_reaction_data_biased(biased):
    r"""
    This is the most comprehensive test for biasdness predict_all
    """
    # cell_X_metadata = pd.read_table("tests/data/fake/fake_metadata.csv", sep=",")
    reaction_X_cell_df = pd.read_csv("tests/data/fake/fake_reactions.csv", delimiter=',')
    reaction_X_cell_df.rename(columns={'Unnamed: 0': 'reaction'}, inplace=True)
    reaction_X_cell_df.set_index('reaction', inplace=True)
    # reaction_names = pd.DataFrame({'reaction_name': reaction_X_cell_df.index})
    cell_X_reaction_df = reaction_X_cell_df.transpose()
    del reaction_X_cell_df
    X = np.array(cell_X_reaction_df)
    R, C = X.shape

    np.random.seed(1)
    random_matrix_indices = get_list_of_random_matrix_indices(X.shape[0], X.shape[1], 0.5)
    observations = observe_entries(X, random_matrix_indices)
    X_observed = matrix_from_observations(observations, nrows=X.shape[0], ncols=X.shape[1])
    mf_model = MatrixFactorizationModel(
        n_factors=1,
        reg_all=0.0,
        n_epochs=100,
        lr_all=0.01,
        verbose=False,
        biased=biased)
    mf_model.fit(observations, nrows=X.shape[0], ncols=X.shape[1])
    X_completion = mf_model.impute_all(X_observed)
    X_completed_with_predict = X_observed.copy()
    for r in range(R):
        for c in range(C):
            if np.isnan(X_observed[r, c]):
                X_completed_with_predict[r, c] = mf_model.predict(r, c)
    np.testing.assert_almost_equal(X_completion, X_completed_with_predict)
    unobserved_indices = np.where(np.isnan(X_observed))
    X_completion_with_predict_all = X_observed.copy()
    X_completion_with_predict_all[unobserved_indices] = mf_model.predict_all()[unobserved_indices]
    np.testing.assert_almost_equal(X_completion, X_completion_with_predict_all)
    np.testing.assert_almost_equal(X_completed_with_predict, X_completion_with_predict_all)

    X_predict_all = mf_model.predict_all()
    X_predict_in_loop = np.zeros_like(X, dtype=float)
    print(mf_model.predict(0, 0))
    for r in range(R):
        for c in range(C):
            X_predict_in_loop[r, c] = mf_model.predict(r, c)
    np.testing.assert_almost_equal(X_predict_all, X_predict_in_loop)
