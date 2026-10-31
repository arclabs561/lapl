//! Accuracy of the iterative spectral-embedding paths against exact
//! references: Jacobi on small dense Laplacians and the closed-form spectrum
//! of path graphs, 1 - cos(pi j / (m - 1)) for a path on m nodes.
#![allow(dead_code)]

use lapl::{
    normalized_laplacian, spectral_embedding, spectral_embedding_with_diagnostics,
    symmetric_eigenvalues, SpectralEmbeddingConfig,
};
use ndarray::{Array1, Array2};

fn path_adj(n: usize) -> Array2<f64> {
    let mut a = Array2::zeros((n, n));
    for i in 0..n - 1 {
        a[[i, i + 1]] = 1.0;
        a[[i + 1, i]] = 1.0;
    }
    a
}

/// Two disjoint paths of `m` nodes each: nodes `0..m` and `m..2m`.
fn two_paths_edges(m: usize) -> Vec<(usize, usize, f64)> {
    let mut edges = Vec::new();
    for c in 0..2 {
        for i in 0..m - 1 {
            edges.push((c * m + i, c * m + i + 1, 1.0));
        }
    }
    edges
}

fn dense_from_edges(n: usize, edges: &[(usize, usize, f64)]) -> Array2<f64> {
    let mut a = Array2::zeros((n, n));
    for &(u, v, w) in edges {
        a[[u, v]] = w;
        a[[v, u]] = w;
    }
    a
}

fn random_edges(n: usize, p_percent: u64, seed: u64) -> Vec<(usize, usize, f64)> {
    let mut state = seed;
    let mut rnd = |k: u64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) % k
    };
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if rnd(100) < p_percent {
                edges.push((i, j, 1.0));
            }
        }
    }
    edges
}

fn path_spectrum(m: usize) -> Vec<f64> {
    (0..m)
        .map(|j| 1.0 - (std::f64::consts::PI * j as f64 / (m - 1) as f64).cos())
        .collect()
}

/// Rayleigh quotient of each returned column against L_sym, compared with
/// the exact eigenvalues `exact[offset..]`.
fn check_rq(lap: &Array2<f64>, emb: &Array2<f64>, exact: &[f64], offset: usize, label: &str) {
    for c in 0..emb.ncols() {
        let x: Array1<f64> = emb.column(c).to_owned();
        let rq = x.dot(&lap.dot(&x)) / x.dot(&x);
        let want = exact[c + offset];
        assert!(
            (rq - want).abs() < 1e-3 * (1.0 + want) + 1e-4,
            "{label}: column {c} Rayleigh quotient {rq:.6} vs eigenvalue {want:.6}"
        );
    }
}

/// Independent residual check: ||L u - (u'Lu) u|| for each unit column.
fn max_residual(lap: &Array2<f64>, emb: &Array2<f64>) -> f64 {
    let mut worst = 0.0f64;
    for c in 0..emb.ncols() {
        let mut x: Array1<f64> = emb.column(c).to_owned();
        let norm = x.dot(&x).sqrt();
        x /= norm;
        let lx = lap.dot(&x);
        let theta = x.dot(&lx);
        let r = &lx - &(&x * theta);
        worst = worst.max(r.dot(&r).sqrt());
    }
    worst
}

/// Rows of a two-component embedding: rows within a component must point the
/// same way and rows across components must be orthogonal (after row
/// normalization), which is what spectral clustering relies on.
fn assert_separates_components(emb: &Array2<f64>, m: usize, label: &str) {
    let unit = |i: usize| {
        let r = emb.row(i).to_owned();
        let n = r.dot(&r).sqrt();
        r / n
    };
    for (i, j) in [(0, m - 1), (m / 2, 1), (m, 2 * m - 1), (m + 3, m + m / 2)] {
        let c = unit(i).dot(&unit(j));
        assert!(
            c > 0.99,
            "{label}: rows {i},{j} in one component: cos {c:.4}"
        );
    }
    for (i, j) in [(0, m), (m / 2, m + m / 2), (m - 1, 2 * m - 1)] {
        let c = unit(i).dot(&unit(j)).abs();
        assert!(
            c < 0.05,
            "{label}: rows {i},{j} across components: |cos| {c:.4}"
        );
    }
}

fn raw_cfg(skip_first: bool) -> SpectralEmbeddingConfig {
    SpectralEmbeddingConfig {
        skip_first,
        row_normalize: false,
        ..Default::default()
    }
}

#[test]
fn path_spectrum_closed_form_matches_jacobi() {
    let m = 20;
    let exact = symmetric_eigenvalues(&normalized_laplacian(&path_adj(m)), 1e-13, 200_000).unwrap();
    for (a, b) in exact.iter().zip(path_spectrum(m)) {
        assert!((a - b).abs() < 1e-9, "{a} vs {b}");
    }
}

// Path P100 is above the default Jacobi threshold (64). Without `faer` this
// runs the iterative dense path.
#[test]
fn dense_embedding_large_n_matches_eigenvectors() {
    let adj = path_adj(100);
    let lap = normalized_laplacian(&adj);
    let emb = spectral_embedding(&adj, 3, &raw_cfg(true)).unwrap();
    let exact = symmetric_eigenvalues(&lap, 1e-12, 500_000).unwrap();
    check_rq(&lap, &emb, &exact, 1, "spectral_embedding(P100)");
    assert!(max_residual(&lap, &emb) < 1e-6);
}

// Two disconnected paths of 300 nodes: n = 600 is above the `faer` dense
// threshold too, so every feature set runs an iterative solver here. The
// zero eigenvalue is double, which a single start vector cannot resolve.
#[test]
fn dense_embedding_two_components_n600() {
    let m = 300;
    let adj = dense_from_edges(2 * m, &two_paths_edges(m));
    let lap = normalized_laplacian(&adj);
    let emb = spectral_embedding(&adj, 2, &raw_cfg(false)).unwrap();
    let res = max_residual(&lap, &emb);
    assert!(res < 1e-6, "max residual {res:e}");
    check_rq(&lap, &emb, &[0.0, 0.0], 0, "two paths n=600");
    assert_separates_components(&emb, m, "two paths n=600");
}

// The iterative path reports non-convergence instead of returning an
// unconverged embedding: with no restarts allowed, P100's first Krylov basis
// (64 vectors) cannot resolve its small eigenvalue gaps.
#[cfg(not(feature = "faer"))]
#[test]
fn dense_embedding_reports_non_convergence() {
    let adj = path_adj(100);
    let mut cfg = raw_cfg(true);
    cfg.iters = 0;
    let (emb, diag) = spectral_embedding_with_diagnostics(&adj, 3, &cfg).unwrap();
    assert!(!diag.converged);
    assert_eq!(diag.restarts, 0);
    assert!(diag.max_residual() > diag.tolerance);
    assert_eq!(emb.dim(), (100, 3));
    assert!(matches!(
        spectral_embedding(&adj, 3, &cfg),
        Err(lapl::Error::Backend(_))
    ));

    let (_, diag) = spectral_embedding_with_diagnostics(&adj, 3, &raw_cfg(true)).unwrap();
    assert!(diag.converged, "{diag:?}");
    assert!(diag.restarts > 0);
    let exact = path_spectrum(100);
    for (got, want) in diag.eigenvalues.iter().zip(&exact[1..]) {
        assert!((got - want).abs() < 1e-9, "{got} vs {want}");
    }
}

#[test]
fn jacobi_path_diagnostics_are_converged() {
    let adj = path_adj(30);
    let (_, diag) = spectral_embedding_with_diagnostics(&adj, 2, &raw_cfg(true)).unwrap();
    assert!(diag.converged, "{diag:?}");
    assert_eq!(diag.restarts, 0);
    assert_eq!(diag.residuals.len(), 2);
}

#[cfg(feature = "sparse")]
mod sparse {
    use super::*;
    use lapl::sparse::{
        spectral_embedding_sparse, spectral_embedding_sparse_with_diagnostics, CsrAdjacency,
    };

    #[test]
    fn sparse_embedding_reports_non_convergence() {
        let n = 100;
        let edges: Vec<(usize, usize, f64)> = (0..n - 1).map(|i| (i, i + 1, 1.0)).collect();
        let csr = CsrAdjacency::from_undirected_edges(n, &edges).unwrap();
        let mut cfg = raw_cfg(true);
        cfg.iters = 0;
        let (_, diag) = spectral_embedding_sparse_with_diagnostics(&csr, 3, &cfg).unwrap();
        assert!(!diag.converged);
        assert!(diag.max_residual() > diag.tolerance);
        assert!(matches!(
            spectral_embedding_sparse(&csr, 3, &cfg),
            Err(lapl::Error::Backend(_))
        ));
    }

    #[test]
    fn sparse_embedding_matches_eigenvectors() {
        let n = 100;
        let edges: Vec<(usize, usize, f64)> = (0..n - 1).map(|i| (i, i + 1, 1.0)).collect();
        let csr = CsrAdjacency::from_undirected_edges(n, &edges).unwrap();
        let emb = spectral_embedding_sparse(&csr, 3, &raw_cfg(true)).unwrap();
        let lap = normalized_laplacian(&path_adj(n));
        check_rq(
            &lap,
            &emb,
            &path_spectrum(n),
            1,
            "spectral_embedding_sparse(P100)",
        );
        assert!(max_residual(&lap, &emb) < 1e-6);
    }

    // An ordinary 46-node random graph (below the dense Jacobi threshold, so
    // only the sparse path is iterative).
    #[test]
    fn sparse_embedding_small_random_graph() {
        let n = 46;
        let edges = random_edges(n, 25, 7);
        let adj = dense_from_edges(n, &edges);
        let csr = CsrAdjacency::from_undirected_edges(n, &edges).unwrap();
        let emb = spectral_embedding_sparse(&csr, 3, &raw_cfg(true)).unwrap();
        let lap = normalized_laplacian(&adj);
        let exact = symmetric_eigenvalues(&lap, 1e-12, 500_000).unwrap();
        check_rq(
            &lap,
            &emb,
            &exact,
            1,
            "spectral_embedding_sparse(G(46, 0.25))",
        );
    }

    #[test]
    fn sparse_embedding_two_components_n800() {
        let m = 400;
        let edges = two_paths_edges(m);
        let csr = CsrAdjacency::from_undirected_edges(2 * m, &edges).unwrap();
        let emb = spectral_embedding_sparse(&csr, 3, &raw_cfg(false)).unwrap();
        let spectrum = path_spectrum(m);
        // Union of both paths' spectra: 0, 0, lambda_1, lambda_1, ...
        let exact = [0.0, 0.0, spectrum[1]];
        let lap = normalized_laplacian(&dense_from_edges(2 * m, &edges));
        check_rq(&lap, &emb, &exact, 0, "sparse two paths n=800");
        let res = max_residual(&lap, &emb);
        assert!(res < 1e-6, "max residual {res:e}");
        assert_separates_components(
            &emb.slice(ndarray::s![.., 0..2]).to_owned(),
            m,
            "sparse two paths",
        );
    }

    // A denser random graph checked against the exact dense solver.
    #[cfg(feature = "faer")]
    #[test]
    fn sparse_embedding_matches_dense_on_random_graph_n400() {
        let n = 400;
        let edges = random_edges(n, 2, 11);
        let adj = dense_from_edges(n, &edges);
        // Drop isolated nodes from the comparison by requiring connectivity.
        assert!(lapl::is_connected(&adj), "fixture should be connected");
        let lap = normalized_laplacian(&adj);
        let csr = CsrAdjacency::from_undirected_edges(n, &edges).unwrap();
        let sparse = spectral_embedding_sparse(&csr, 4, &raw_cfg(true)).unwrap();
        // n = 400 with `faer` is the exact dense eigendecomposition.
        let dense = spectral_embedding(&adj, 4, &raw_cfg(true)).unwrap();
        for c in 0..4 {
            let rq = |e: &Array2<f64>| {
                let x = e.column(c).to_owned();
                x.dot(&lap.dot(&x)) / x.dot(&x)
            };
            let (a, b) = (rq(&sparse), rq(&dense));
            assert!((a - b).abs() < 1e-8, "column {c}: sparse {a} vs dense {b}");
        }
        assert!(max_residual(&lap, &sparse) < 1e-6);
    }
}
