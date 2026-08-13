//! Analytic oracles for iterative spectral embeddings.
//!
//! Eigenvectors in repeated eigenspaces are not unique, so these tests compare
//! subspaces through their projection error rather than comparing columns.

use lapl::{normalized_laplacian, spectral_embedding, SpectralEmbeddingConfig};
use ndarray::{Array2, ArrayView2};

#[cfg(feature = "sparse")]
use lapl::sparse::{spectral_embedding_sparse, CsrAdjacency};

fn path_adjacency(n: usize) -> Array2<f64> {
    let mut adj = Array2::zeros((n, n));
    for i in 0..(n - 1) {
        adj[[i, i + 1]] = 1.0;
        adj[[i + 1, i]] = 1.0;
    }
    adj
}

fn cycle_adjacency(n: usize) -> Array2<f64> {
    let mut adj = path_adjacency(n);
    adj[[0, n - 1]] = 1.0;
    adj[[n - 1, 0]] = 1.0;
    adj
}

fn path_mode(n: usize, mode: usize) -> Array2<f64> {
    let mut v = Array2::zeros((n, 1));
    for j in 0..n {
        let degree: f64 = if j == 0 || j + 1 == n { 1.0 } else { 2.0 };
        v[[j, 0]] =
            degree.sqrt() * (std::f64::consts::PI * mode as f64 * j as f64 / (n - 1) as f64).cos();
    }
    normalize_columns(v)
}

fn cycle_fiedler_space(n: usize) -> Array2<f64> {
    let mut basis = Array2::zeros((n, 2));
    for j in 0..n {
        let angle = 2.0 * std::f64::consts::PI * j as f64 / n as f64;
        basis[[j, 0]] = angle.cos();
        basis[[j, 1]] = angle.sin();
    }
    normalize_columns(basis)
}

fn normalize_columns(mut basis: Array2<f64>) -> Array2<f64> {
    for mut column in basis.columns_mut() {
        let norm = column.dot(&column).sqrt();
        column.mapv_inplace(|x| x / norm);
    }
    basis
}

fn projection_error(actual: ArrayView2<'_, f64>, expected: ArrayView2<'_, f64>) -> f64 {
    let projected = actual.dot(&actual.t().dot(&expected));
    let residual = &expected - &projected;
    residual.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn ritz_residual(lap: &Array2<f64>, basis: ArrayView2<'_, f64>) -> f64 {
    let lb = lap.dot(&basis);
    let projected = basis.dot(&basis.t().dot(&lb));
    (&lb - &projected).iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn iterative_config() -> SpectralEmbeddingConfig {
    SpectralEmbeddingConfig {
        iters: 500,
        jacobi_max_n: 0,
        jacobi_tol: 1e-12,
        jacobi_max_sweeps: 50_000,
        skip_first: true,
        row_normalize: false,
    }
}

#[test]
fn iterative_path_recovers_first_nontrivial_mode() {
    let n = 16;
    let adj = path_adjacency(n);
    let lap = normalized_laplacian(&adj);
    let actual = spectral_embedding(&adj, 1, &iterative_config()).unwrap();
    let expected = path_mode(n, 1);

    let subspace_error = projection_error(actual.view(), expected.view());
    let residual = ritz_residual(&lap, actual.view());
    assert!(
        subspace_error < 1e-7,
        "subspace error {subspace_error}, Ritz residual {residual}"
    );
    assert!(residual < 1e-7, "Ritz residual {residual}");
}

#[test]
fn iterative_cycle_recovers_repeated_fiedler_space() {
    let n = 16;
    let adj = cycle_adjacency(n);
    let lap = normalized_laplacian(&adj);
    let actual = spectral_embedding(&adj, 2, &iterative_config()).unwrap();
    let expected = cycle_fiedler_space(n);

    assert!(projection_error(actual.view(), expected.view()) < 1e-7);
    assert!(ritz_residual(&lap, actual.view()) < 1e-7);
}

#[test]
fn disconnected_embedding_preserves_the_full_kernel_when_not_skipping() {
    let n = 16;
    let mut adj = Array2::zeros((n, n));
    for i in 0..7 {
        adj[[i, i + 1]] = 1.0;
        adj[[i + 1, i]] = 1.0;
    }
    for i in 8..16 {
        let next = 8 + (i - 7) % 8;
        adj[[i, next]] = 1.0;
        adj[[next, i]] = 1.0;
    }

    let mut expected = Array2::zeros((n, 2));
    for i in 0..8 {
        let degree: f64 = if i == 0 || i == 7 { 1.0 } else { 2.0 };
        expected[[i, 0]] = degree.sqrt();
    }
    for i in 8..16 {
        expected[[i, 1]] = 2.0_f64.sqrt();
    }
    expected = normalize_columns(expected);

    let mut cfg = iterative_config();
    cfg.skip_first = false;
    let actual = spectral_embedding(&adj, 2, &cfg).unwrap();
    let lap = normalized_laplacian(&adj);

    let subspace_error = projection_error(actual.view(), expected.view());
    let residual = ritz_residual(&lap, actual.view());
    assert!(
        subspace_error < 1e-7,
        "subspace error {subspace_error}, Ritz residual {residual}"
    );
    assert!(residual < 1e-7, "Ritz residual {residual}");
}

#[cfg(feature = "sparse")]
fn path_edges(n: usize) -> Vec<(usize, usize, f64)> {
    (0..(n - 1)).map(|i| (i, i + 1, 1.0)).collect()
}

#[cfg(feature = "sparse")]
#[test]
fn sparse_path_recovers_first_nontrivial_mode() {
    let n = 16;
    let sparse = CsrAdjacency::from_undirected_edges(n, &path_edges(n)).unwrap();
    let actual = spectral_embedding_sparse(&sparse, 1, &iterative_config()).unwrap();
    let expected = path_mode(n, 1);
    let lap = normalized_laplacian(&path_adjacency(n));

    assert!(projection_error(actual.view(), expected.view()) < 1e-7);
    assert!(ritz_residual(&lap, actual.view()) < 1e-7);
}

#[cfg(feature = "sparse")]
#[test]
fn sparse_cycle_recovers_repeated_fiedler_space() {
    let n = 16;
    let mut edges = path_edges(n);
    edges.push((n - 1, 0, 1.0));
    let sparse = CsrAdjacency::from_undirected_edges(n, &edges).unwrap();
    let actual = spectral_embedding_sparse(&sparse, 2, &iterative_config()).unwrap();
    let expected = cycle_fiedler_space(n);
    let lap = normalized_laplacian(&cycle_adjacency(n));

    assert!(projection_error(actual.view(), expected.view()) < 1e-7);
    assert!(ritz_residual(&lap, actual.view()) < 1e-7);
}

#[cfg(feature = "faer")]
#[test]
fn faer_partial_solver_returns_the_first_nontrivial_path_mode() {
    // n > 512 selects the partial Krylov--Schur backend.
    let n = 513;
    let adj = path_adjacency(n);
    let lap = normalized_laplacian(&adj);
    let cfg = SpectralEmbeddingConfig {
        jacobi_max_n: 0,
        row_normalize: false,
        ..Default::default()
    };
    let actual = spectral_embedding(&adj, 1, &cfg).unwrap();
    let expected = path_mode(n, 1);

    let subspace_error = projection_error(actual.view(), expected.view());
    let residual = ritz_residual(&lap, actual.view());
    assert!(subspace_error < 1e-6, "subspace error {subspace_error}");
    assert!(residual < 1e-7, "Ritz residual {residual}");
}
