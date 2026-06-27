//! Closed-form spectral oracles for the graph Laplacian.
//!
//! These integration tests pin the crate's Laplacian construction and its
//! eigensolver against spectra that are known analytically, never against
//! another solver. The graphs are small, regular, and have circulant or
//! path structure, so every eigenpair is hand-derivable.
//!
//! Two invariants are exercised:
//!
//! 1. **Full-spectrum eigenpair oracle.** For a symmetric `n x n` matrix, any
//!    set of `n` mutually orthogonal vectors `v_i` satisfying `L v_i = λ_i v_i`
//!    *is* the eigendecomposition (spectral theorem: `n` eigenvalues with
//!    multiplicity, `n` orthonormal eigenvectors determine the matrix). So
//!    exhibiting the analytic orthogonal eigenbasis and checking the residual
//!    `‖L v − λ v‖` verifies the entire closed-form spectrum without running an
//!    eigensolver in the test, and without reimplementing one.
//!
//! 2. **Kernel / connectivity invariant.** `L = D − A` is positive
//!    semidefinite; the all-ones vector is always in its kernel (row sums are
//!    zero), and the dimension of the kernel equals the number of connected
//!    components (Chung, *Spectral Graph Theory*; von Luxburg 2007). The real
//!    eigensolver (`spectral_embedding`) is then run to recover the algebraic
//!    connectivity `λ_2` and confirm it matches the closed form.

use lapl::{
    adjacency_to_laplacian, laplacian_quadratic_form, normalized_laplacian, spectral_embedding,
    SpectralEmbeddingConfig,
};
use ndarray::{array, Array1, Array2};

/// Residual of a candidate eigenpair: `‖L v − λ v‖_2`.
///
/// Pure linear algebra against the matrix the crate built; this is not an
/// eigensolver, it only checks a claimed (eigenvalue, eigenvector) pair.
fn eigenpair_residual(lap: &Array2<f64>, v: &Array1<f64>, lambda: f64) -> f64 {
    let lv = lap.dot(v);
    let resid = &lv - &(v * lambda);
    resid.dot(&resid).sqrt()
}

/// Rayleigh quotient `v^T L v / v^T v`. For an eigenvector this equals the
/// eigenvalue, so it lets the eigensolver's output be compared to the closed
/// form even though the public API returns eigenvectors, not eigenvalues.
fn rayleigh_quotient(lap: &Array2<f64>, v: &Array1<f64>) -> f64 {
    let denom = v.dot(v);
    laplacian_quadratic_form(lap, v) / denom
}

/// Path graph P3 (`0 — 1 — 2`), unnormalized Laplacian.
///
/// ```text
///         [  1  -1   0 ]
/// L = D-A=[ -1   2  -1 ]
///         [  0  -1   1 ]
/// ```
///
/// Closed-form spectrum (characteristic polynomial `λ(λ−1)(λ−3)`):
///   trace = 4 = 0 + λ2 + λ3, sum of principal 2×2 minors = 3 = λ2·λ3,
///   so {λ2, λ3} are the roots of t² − 4t + 3 = (t−1)(t−3) ⇒ spectrum = {0, 1, 3}.
/// Orthogonal eigenvectors:
///   λ=0 → (1, 1, 1)   (kernel: connected graph ⇒ one-dimensional)
///   λ=1 → (1, 0, −1)
///   λ=3 → (1, −2, 1)
#[test]
fn p3_unnormalized_laplacian_has_spectrum_0_1_3() {
    let adj = array![[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]];
    let lap = adjacency_to_laplacian(&adj);

    let pairs: [(Array1<f64>, f64); 3] = [
        (array![1.0, 1.0, 1.0], 0.0),
        (array![1.0, 0.0, -1.0], 1.0),
        (array![1.0, -2.0, 1.0], 3.0),
    ];

    // Each analytic pair must satisfy L v = λ v.
    for (v, lambda) in &pairs {
        let r = eigenpair_residual(&lap, v, *lambda);
        assert!(r < 1e-12, "P3 eigenpair λ={lambda} residual {r} too large");
    }

    // The eigenvectors are mutually orthogonal, so they form a full basis and
    // {0,1,3} is the entire spectrum (not just three of n eigenvalues).
    for a in 0..3 {
        for b in (a + 1)..3 {
            let dot = pairs[a].0.dot(&pairs[b].0);
            assert!(
                dot.abs() < 1e-12,
                "P3 eigenvectors {a},{b} not orthogonal: {dot}"
            );
        }
    }
}

/// Cycle graph C4 (`0 — 1 — 2 — 3 — 0`), 2-regular ⇒ circulant.
///
/// The adjacency is circulant, so its eigenvectors are the DFT (Fourier) modes
/// and its eigenvalues are `2·cos(2πk/4)` for k = 0..3, i.e. {2, 0, 0, −2}.
///
/// Unnormalized Laplacian L = 2I − A (degree 2 everywhere), so its eigenvalues
/// are `2 − {2, 0, 0, −2} = {0, 2, 2, 4}` on the same Fourier eigenvectors:
///   λ=0 → (1,  1,  1,  1)
///   λ=2 → (1,  0, −1,  0)   and   (0, 1, 0, −1)   (multiplicity 2)
///   λ=4 → (1, −1,  1, −1)
#[test]
fn c4_unnormalized_laplacian_has_spectrum_0_2_2_4() {
    let adj = array![
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
    ];
    let lap = adjacency_to_laplacian(&adj);

    let pairs: [(Array1<f64>, f64); 4] = [
        (array![1.0, 1.0, 1.0, 1.0], 0.0),
        (array![1.0, 0.0, -1.0, 0.0], 2.0),
        (array![0.0, 1.0, 0.0, -1.0], 2.0),
        (array![1.0, -1.0, 1.0, -1.0], 4.0),
    ];

    for (v, lambda) in &pairs {
        let r = eigenpair_residual(&lap, v, *lambda);
        assert!(r < 1e-12, "C4 eigenpair λ={lambda} residual {r} too large");
    }

    // Mutual orthogonality ⇒ these four pairs are the complete spectrum.
    for a in 0..4 {
        for b in (a + 1)..4 {
            let dot = pairs[a].0.dot(&pairs[b].0);
            assert!(
                dot.abs() < 1e-12,
                "C4 eigenvectors {a},{b} not orthogonal: {dot}"
            );
        }
    }
}

/// Normalized Laplacian of C4 and the eigensolver's algebraic connectivity.
///
/// C4 is 2-regular, so `L_sym = I − A/2` and its spectrum is
/// `1 − {2, 0, 0, −2}/2 = {0, 1, 1, 2}`. The algebraic connectivity is the
/// second-smallest eigenvalue λ_2 = 1.
///
/// This test runs the *real* eigensolver (`spectral_embedding`, Jacobi path for
/// n ≤ 64): it returns the Fiedler eigenvector (smallest non-trivial), whose
/// Rayleigh quotient against `L_sym` must equal the closed-form λ_2 = 1.
#[test]
fn c4_normalized_fiedler_value_matches_closed_form() {
    let adj = array![
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
    ];
    let l_sym = normalized_laplacian(&adj);

    // Closed-form check on the analytic Fiedler vector (eigenvalue-1 subspace).
    let fiedler_analytic = array![1.0, 0.0, -1.0, 0.0];
    assert!(
        eigenpair_residual(&l_sym, &fiedler_analytic, 1.0) < 1e-12,
        "analytic Fiedler vector is not an eigenvector of L_sym with λ=1"
    );

    // Now the eigensolver: skip the trivial constant vector, keep the raw
    // eigenvector (no row-normalization, which would destroy the eigenpair).
    let cfg = SpectralEmbeddingConfig {
        skip_first: true,
        row_normalize: false,
        ..Default::default()
    };
    let emb = spectral_embedding(&adj, 1, &cfg).unwrap();
    assert_eq!(emb.dim(), (4, 1));

    let fiedler = emb.column(0).to_owned();
    let lambda2 = rayleigh_quotient(&l_sym, &fiedler);
    assert!(
        (lambda2 - 1.0).abs() < 1e-8,
        "eigensolver Fiedler value {lambda2} != closed-form λ_2 = 1"
    );
}

/// Kernel dimension equals the number of connected components.
///
/// Two disjoint edges {0—1} and {2—3} form a 2-component graph. The unnormalized
/// Laplacian is block-diagonal; each block is the P2 Laplacian `[[1,−1],[−1,1]]`
/// with kernel spanned by its component indicator. So the kernel of L is
/// two-dimensional, spanned by the indicators (1,1,0,0) and (0,0,1,1) — exactly
/// the number of components. The global all-ones vector is their sum, also in
/// the kernel.
#[test]
fn disconnected_graph_kernel_dim_equals_component_count() {
    let adj = array![
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ];
    let lap = adjacency_to_laplacian(&adj);

    // Both component indicators are in the kernel (λ = 0).
    let ind_a = array![1.0, 1.0, 0.0, 0.0];
    let ind_b = array![0.0, 0.0, 1.0, 1.0];
    assert!(eigenpair_residual(&lap, &ind_a, 0.0) < 1e-12);
    assert!(eigenpair_residual(&lap, &ind_b, 0.0) < 1e-12);

    // They are linearly independent (orthogonal), so kernel dim ≥ 2 = #components.
    assert!(ind_a.dot(&ind_b).abs() < 1e-12);

    // The all-ones vector is in the kernel too (it is ind_a + ind_b).
    let ones = array![1.0, 1.0, 1.0, 1.0];
    assert!(eigenpair_residual(&lap, &ones, 0.0) < 1e-12);
}

/// Connected graph has a one-dimensional kernel: all-ones spans it, and the
/// algebraic connectivity λ_2 is strictly positive.
///
/// Triangle K3 (3-regular-ish, every node degree 2) has normalized-Laplacian
/// spectrum {0, 3/2, 3/2}: `L_sym = I − A/2`, adjacency eigenvalues of K3 are
/// {2, −1, −1}, so `1 − {2,−1,−1}/2 = {0, 3/2, 3/2}`. The eigensolver must
/// return a Fiedler vector with Rayleigh quotient 3/2 > 0, proving the kernel is
/// exactly the constant direction (one zero eigenvalue ⇒ one component).
#[test]
fn connected_graph_has_positive_algebraic_connectivity() {
    let adj = array![[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]];
    let l_sym = normalized_laplacian(&adj);

    // all-ones is in the kernel.
    let ones = array![1.0, 1.0, 1.0];
    assert!(eigenpair_residual(&l_sym, &ones, 0.0) < 1e-12);

    let cfg = SpectralEmbeddingConfig {
        skip_first: true,
        row_normalize: false,
        ..Default::default()
    };
    let emb = spectral_embedding(&adj, 1, &cfg).unwrap();
    let fiedler = emb.column(0).to_owned();
    let lambda2 = rayleigh_quotient(&l_sym, &fiedler);

    assert!(
        (lambda2 - 1.5).abs() < 1e-8,
        "K3 algebraic connectivity {lambda2} != closed-form 3/2"
    );
    assert!(lambda2 > 1e-6, "connected graph must have λ_2 > 0");
}
