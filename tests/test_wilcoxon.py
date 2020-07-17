import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from turbo_mc.wilcoxon import wilcoxon_test, wilcoxon_test_naive, wilcoxon_score,\
    get_DE_genes


def test_wilcoxon_fast_against_naive_implementation():
    for repetition in range(10):
        np.random.seed(repetition)
        N = 100
        xs = range(repetition + 1)
        ys = [0, 1]
        x = np.random.choice(xs, N)
        y = np.random.choice(ys, N)
        assert(wilcoxon_test(x, y) == wilcoxon_test_naive(x, y))


@pytest.mark.slow
def test_null_distribution_is_correct():
    np.random.seed(1)
    N = 290
    xs = range(2)
    x = np.random.choice(xs, N)
    # ys = [0, 1]
    # y = np.random.choice(ys, N)
    y = np.array([0] * (N - 10) + [1] * 10)
    correctly_ordered_pairss = []
    mus = []
    sigmas = []
    sigmas_corrected = []
    for repetition in range(3000):
        x_i = np.random.permutation(x)
        correctly_ordered_pairs, mu, sigma, sigma_corrected =\
            wilcoxon_test(x_i, y, return_null_distribution=True)
        correctly_ordered_pairss.append(correctly_ordered_pairs)
        mus.append(mu)
        sigmas.append(sigma)
        sigmas_corrected.append(sigma_corrected)
    plt.hist(correctly_ordered_pairss, bins=50)
    assert(len(set(mus)) == 1)
    assert(len(set(sigmas)) == 1)
    empirical_mean = np.mean(correctly_ordered_pairss)
    plt.vlines(x=mus[0], ymin=0, ymax=plt.ylim()[1])
    plt.vlines(x=empirical_mean, ymin=0, ymax=plt.ylim()[1], colors='r')
    empirical_std = np.std(correctly_ordered_pairss)
    plt.vlines(x=[mus[0] - sigmas_corrected[0], mus[0] + sigmas_corrected[0]], ymin=0, ymax=plt.ylim()[1])
    plt.vlines(x=[empirical_mean - empirical_std,
                  empirical_mean + empirical_std], ymin=0, ymax=plt.ylim()[1], colors='r')
    plt.vlines(x=[mus[0] - sigmas[0], mus[0] + sigmas[0]], ymin=0, ymax=plt.ylim()[1], color='y')
    assert(np.abs((empirical_mean - mus[0]) / mus[0]) < 1e-1)
    assert(np.abs((empirical_std - sigmas_corrected[0]) / sigmas_corrected[0]) < 1e-1)
    assert(np.abs((empirical_std - sigmas[0]) / sigmas[0]) > 1e-1)


@pytest.mark.slow
def test_wilcoxon_score():
    np.random.seed(1)
    N = 290
    xs = range(2)
    x = np.random.choice(xs, N)
    # ys = [0, 1]
    # y = np.random.choice(ys, N)
    y = np.array([0] * (N - int(N / 4)) + [1] * int(N / 4))
    wilcoxon_scores = []
    for repetition in range(1000):
        x_i = np.random.permutation(x)
        curr_wilcoxon_score = wilcoxon_score(x_i, y)
        wilcoxon_scores.append(curr_wilcoxon_score)
    plt.hist(wilcoxon_scores)
    wilcoxon_scores = np.array(wilcoxon_scores)
    print(f"{np.mean(np.abs(wilcoxon_scores) < 1.0)} ~ 0.68")
    print(f"{np.mean(np.abs(wilcoxon_scores) < 2.0)} ~ 0.96")
    assert(np.abs(np.mean(np.abs(wilcoxon_scores) < 1.0) - 0.68) < 0.02)
    assert(np.abs(np.mean(np.abs(wilcoxon_scores) < 2.0) - 0.96) < 0.02)


@pytest.mark.parametrize("normalize_cell_counts", [True, False])
def test_get_DE_genes(normalize_cell_counts):
    gene_expression_matrix_df = pd.read_table("tests/data/fake/fake_reactions.csv", sep=',')
    gene_expression_matrix_df.rename(columns={'Unnamed: 0': 'reaction_id'}, inplace=True)
    gene_expression_matrix_df.set_index('reaction_id', inplace=True)
    cell_X_metadata_df = pd.read_table("tests/data/fake/fake_metadata.csv", sep=',')
    cell_X_metadata_df.set_index('cell_id', inplace=True)
    res_df = get_DE_genes(
        gene_expression_matrix_df,
        cell_X_metadata_df,
        metadata_col='cell_type',
        id0='Th17p',
        id1='Th17n',
        pseudocounts=1e-16,
        normalize_cell_counts=normalize_cell_counts,
        scale_factor=1e6
    )
    print(res_df)
    assert(True)


def test_get_DE_genes_with_NaNs():
    r"""
    Running DE with NaN's shouldn't change results.
    """
    # Without nans:
    gene_expression_matrix_df = pd.read_table("tests/data/fake/fake_reactions.csv", sep=',')
    gene_expression_matrix_df.rename(columns={'Unnamed: 0': 'reaction_id'}, inplace=True)
    gene_expression_matrix_df.set_index('reaction_id', inplace=True)
    cell_X_metadata_df = pd.read_table("tests/data/fake/fake_metadata.csv", sep=',')
    cell_X_metadata_df.set_index('cell_id', inplace=True)
    res_df = get_DE_genes(
        gene_expression_matrix_df,
        cell_X_metadata_df,
        metadata_col='cell_type',
        id0='Th17p',
        id1='Th17n',
        pseudocounts=1e-16,
        normalize_cell_counts=False,
        scale_factor=1e6
    )
    res_df_no_nans = res_df

    # With nans, and a shuffled index
    gene_expression_matrix_df = pd.read_table("tests/data/fake_with_nans/fake_reactions_nan.csv", sep=',')
    gene_expression_matrix_df.rename(columns={'Unnamed: 0': 'reaction_id'}, inplace=True)
    gene_expression_matrix_df.set_index('reaction_id', inplace=True)
    cell_X_metadata_df = pd.read_table("tests/data/fake_with_nans/fake_metadata_nan.csv", sep=',')
    cell_X_metadata_df.set_index('cell_id', inplace=True)
    res_df = get_DE_genes(
        gene_expression_matrix_df,
        cell_X_metadata_df,
        metadata_col='cell_type',
        id0='Th17p',
        id1='Th17n',
        pseudocounts=1e-16,
        normalize_cell_counts=False,
        scale_factor=1e6
    )
    res_df_nans = res_df

    # Trying to call with normalize_cell_counts=True should raise when there are nans.
    with pytest.raises(ValueError):
        get_DE_genes(
            gene_expression_matrix_df,
            cell_X_metadata_df,
            metadata_col='cell_type',
            id0='Th17p',
            id1='Th17n',
            pseudocounts=1e-16,
            normalize_cell_counts=True,
            scale_factor=1e6
        )

    print(res_df_no_nans.sort_index())
    print(res_df_nans.sort_index())
    assert((res_df_no_nans.sort_index() == res_df_nans.sort_index()).all().all())
