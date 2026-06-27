//! Rosetta correctness fixtures: lapl graph Laplacians and spectral embedding
//! asserted against scipy and numpy.
//!
//! Reference values in `fixtures/rosetta/lapl_spectral.json` come from
//! `gen_lapl.py` (their provenance).
//!
//! Matrix oracles (EXACT, element-wise): adjacency_to_laplacian vs
//! scipy.sparse.csgraph.laplacian(normed=False), normalized_laplacian vs
//! scipy laplacian(normed=True), random_walk_laplacian vs the numpy formula
//! I - D^-1 A.
//!
//! Spectral embedding (SUBSPACE class): eigenvectors are unique only up to sign
//! and, under degeneracy, rotation, so they are NOT compared element-wise.
//! The fixture ships the reference invariant subspace (the k smallest
//! eigenvectors of L_sym from numpy.linalg.eigh) and this test checks that each
//! reference eigenvector lies in the span of lapl's orthonormal
//! spectral_embedding output (projection residual ~ 0). The graph is a complete
//! weighted graph, so the eigenvalues are distinct (gap ~0.15 at k=2) and the
//! smallest-k subspace is well defined.
//!
//! Regenerate the fixture: `uv run tests/fixtures/rosetta/gen_lapl.py`.

use lapl::{
    adjacency_to_laplacian, normalized_laplacian, random_walk_laplacian, spectral_embedding,
    SpectralEmbeddingConfig,
};
use ndarray::{Array1, Array2};
use serde::Deserialize;

const FIXTURE: &str = include_str!("fixtures/rosetta/lapl_spectral.json");

#[derive(Deserialize)]
struct Fixture {
    k: usize,
    adj: Vec<Vec<f64>>,
    expected: Expected,
}

#[derive(Deserialize)]
struct Expected {
    lap_unnormalized: Vec<Vec<f64>>,
    lap_normalized: Vec<Vec<f64>>,
    lap_random_walk: Vec<Vec<f64>>,
    eig_subspace: Vec<Vec<f64>>,
}

fn to_array2(rows: &[Vec<f64>]) -> Array2<f64> {
    let d = rows[0].len();
    let mut a = Array2::zeros((rows.len(), d));
    for (i, r) in rows.iter().enumerate() {
        for (j, &v) in r.iter().enumerate() {
            a[[i, j]] = v;
        }
    }
    a
}

fn mat_close(got: &Array2<f64>, want: &[Vec<f64>], label: &str) {
    assert_eq!(got.nrows(), want.len(), "{label}: row count");
    for (i, row) in want.iter().enumerate() {
        for (j, &w) in row.iter().enumerate() {
            let tol = 1e-9 * (1.0 + w.abs());
            let diff = (got[[i, j]] - w).abs();
            assert!(
                diff <= tol,
                "{label}[{i}][{j}]: lapl={} scipy={w} diff={diff}",
                got[[i, j]]
            );
        }
    }
}

#[test]
fn rosetta_laplacians_match_scipy() {
    let fx: Fixture = serde_json::from_str(FIXTURE).expect("parse rosetta fixture");
    let a = to_array2(&fx.adj);

    mat_close(
        &adjacency_to_laplacian(&a),
        &fx.expected.lap_unnormalized,
        "lap_unnormalized",
    );
    mat_close(
        &normalized_laplacian(&a),
        &fx.expected.lap_normalized,
        "lap_normalized",
    );
    mat_close(
        &random_walk_laplacian(&a),
        &fx.expected.lap_random_walk,
        "lap_random_walk",
    );
}

#[test]
fn rosetta_spectral_embedding_spans_reference_subspace() {
    let fx: Fixture = serde_json::from_str(FIXTURE).expect("parse rosetta fixture");
    let a = to_array2(&fx.adj);

    // skip_first=false keeps the smallest-k eigenvectors (matching the reference);
    // row_normalize=false keeps the raw orthonormal eigenvectors so projection works.
    let cfg = SpectralEmbeddingConfig {
        skip_first: false,
        row_normalize: false,
        ..Default::default()
    };
    let u = spectral_embedding(&a, fx.k, &cfg).expect("spectral embedding");
    assert_eq!(u.ncols(), fx.k);

    // lapl's columns must be orthonormal for the projection test to be exact.
    let gram = u.t().dot(&u);
    for i in 0..fx.k {
        for j in 0..fx.k {
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(
                (gram[[i, j]] - want).abs() < 1e-7,
                "U^T U not identity at [{i}][{j}]: {}",
                gram[[i, j]]
            );
        }
    }

    // Each reference eigenvector must lie in span(U): residual of its projection
    // onto U is ~0. Since both subspaces are k-dimensional and U is orthonormal,
    // span(U_ref) subset span(U) plus equal dimension gives subspace equality.
    for (c, col) in fx.expected.eig_subspace.iter().enumerate() {
        let u_ref = Array1::from(col.clone());
        let coeffs = u.t().dot(&u_ref); // length k
        let proj = u.dot(&coeffs); // length n
        let residual: f64 = (&u_ref - &proj).mapv(|x| x * x).sum().sqrt();
        assert!(
            residual < 1e-6,
            "reference eigenvector {c} not in span(U): residual {residual}"
        );
    }
}
