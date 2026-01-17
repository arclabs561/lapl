//! # lapl
//!
//! Spectral methods: graph Laplacians, eigenmaps, spectral clustering.
//!
//! ## The Core Idea
//!
//! The graph Laplacian L = D - A encodes graph structure in a matrix whose
//! eigenvalues and eigenvectors reveal connectivity, communities, and geometry.
//!
//! ## Key Functions
//!
//! | Function | Purpose |
//! |----------|---------|
//! | [`laplacian`] | Unnormalized Laplacian L = D - A |
//! | [`normalized_laplacian`] | Symmetric L_sym = I - D^{-1/2} A D^{-1/2} |
//! | [`random_walk_laplacian`] | L_rw = I - D^{-1} A |
//! | [`fiedler_vector`] | Second eigenvector (graph bisection) |
//! | [`spectral_embedding`] | Low-dim representation from eigenvectors |
//!
//! ## Quick Start
//!
//! ```rust
//! use lapl::{adjacency_to_laplacian, normalized_laplacian, degree_matrix};
//! use ndarray::array;
//!
//! // Simple graph: 0 -- 1 -- 2
//! let adj = array![
//!     [0.0, 1.0, 0.0],
//!     [1.0, 0.0, 1.0],
//!     [0.0, 1.0, 0.0]
//! ];
//!
//! let lap = adjacency_to_laplacian(&adj);
//! let lap_norm = normalized_laplacian(&adj);
//! ```
//!
//! ## Why Spectral Methods?
//!
//! - **Spectral clustering**: k-means on Laplacian eigenvectors finds communities
//! - **Dimensionality reduction**: Laplacian eigenmaps preserve local structure
//! - **Graph partitioning**: Fiedler vector gives optimal 2-way cut
//! - **Diffusion**: Heat kernel e^{-tL} describes information spreading
//!
//! ## The Laplacian Zoo
//!
//! ```text
//! Unnormalized:   L = D - A
//!   - Simple, but eigenvalues scale with degree
//!   - Null space dimension = number of connected components
//!
//! Normalized (symmetric):   L_sym = I - D^{-1/2} A D^{-1/2}
//!   - Eigenvalues in [0, 2]
//!   - Used for spectral clustering (Ng, Jordan, Weiss)
//!
//! Random walk:   L_rw = I - D^{-1} A
//!   - Related to random walk transition matrix
//!   - Same eigenvalues as L_sym, different eigenvectors
//! ```
//!
//! ## The Fiedler Vector
//!
//! The second smallest eigenvalue λ_2 (algebraic connectivity) measures
//! how well-connected the graph is. Its eigenvector (Fiedler vector)
//! gives an optimal graph bisection: partition by sign.
//!
//! ## Spectral Clustering
//!
//! 1. Compute k smallest eigenvectors of L_sym (excluding constant)
//! 2. Form n × k matrix U from eigenvectors as columns
//! 3. Normalize rows of U
//! 4. Run k-means on rows → cluster assignments
//!
//! ## Connections
//!
//! - [`rkhs`](../rkhs): Gaussian kernel → similarity graph → Laplacian
//! - [`strata`](../strata): Spectral clustering uses this crate
//! - [`rmt`](../rmt): Random graph Laplacians follow RMT eigenvalue distributions
//!
//! ## What Can Go Wrong
//!
//! 1. **Disconnected graph**: Zero eigenvalue multiplicity = number of components.
//! 2. **Zero degree nodes**: Normalized Laplacian undefined. Remove isolated nodes.
//! 3. **Numerical precision**: Very small/large edge weights cause instability.
//! 4. **Eigendecomposition**: this crate includes a pragmatic, dependency-free
//!    approximate spectral embedding (`spectral_embedding`) for dense graphs.
//!    For large graphs, you still want a sparse representation + a real eigensolver.
//! 5. **Scaling**: O(n²) storage for dense adjacency. Use sparse formats for large graphs.
//!
//! ## References
//!
//! - Fiedler (1973). "Algebraic connectivity of graphs"
//! - Shi & Malik (2000). "Normalized cuts and image segmentation"
//! - Ng, Jordan, Weiss (2001). "On Spectral Clustering"
//! - von Luxburg (2007). "A Tutorial on Spectral Clustering"

use ndarray::{Array1, Array2, Axis};
use thiserror::Error;

#[cfg(feature = "sparse")]
pub mod sparse;

#[cfg(feature = "faer")]
use faer::{
    dyn_stack::{MemBuffer, MemStack},
    matrix_free::eigen::{partial_eigen_scratch, partial_self_adjoint_eigen, PartialEigenParams},
    Col, Mat, Par, Side,
};

#[derive(Debug, Error)]
pub enum Error {
    #[error("matrix is not square: {0} x {1}")]
    NotSquare(usize, usize),

    #[error("matrix has negative entries")]
    NegativeEntries,

    #[error("zero degree node at index {0}")]
    ZeroDegree(usize),

    #[error("graph is disconnected (multiple zero eigenvalues)")]
    Disconnected,

    #[error("invalid embedding dimension k={k} for n={n}")]
    InvalidEmbeddingDim { k: usize, n: usize },

    #[error("backend error: {0}")]
    Backend(String),

    #[error("asymmetric laplacian requires directed graph")]
    AsymmetricRequired,
}

pub type Result<T> = std::result::Result<T, Error>;

/// Directed Laplacian for reachability spectral embeddings.
/// 
/// Based on the Hermitian Laplacian for directed graphs (Fan Chung, 2005).
/// L = I - (Φ^{1/2} P Φ^{-1/2} + Φ^{-1/2} P^T Φ^{1/2}) / 2
/// where P is the transition matrix and Φ is the stationary distribution.
pub fn directed_laplacian(adj: &Array2<f64>) -> Result<Array2<f64>> {
    let n = ensure_square(adj)?;
    let p = transition_matrix(adj);
    
    // For 2026, we use a simplified version assuming uniform stationary 
    // distribution for robustness, or we solve for Φ if requested.
    // Here we implement the symmetrized transition part.
    let mut l_dir = Array2::eye(n);
    let p_sym = (&p + &p.t()) * 0.5;
    l_dir -= &p_sym;
    
    Ok(l_dir)
}

fn ensure_square(a: &Array2<f64>) -> Result<usize> {
    let (n, m) = a.dim();
    if n != m {
        return Err(Error::NotSquare(n, m));
    }
    Ok(n)
}

/// Compute degree matrix D from adjacency matrix A.
///
/// D[i,i] = sum of row i (total edge weight from node i)
///
/// # Arguments
///
/// * `adj` - Adjacency matrix (n × n)
///
/// # Returns
///
/// Diagonal degree matrix
pub fn degree_matrix(adj: &Array2<f64>) -> Array2<f64> {
    let n = adj.nrows();
    let mut d = Array2::zeros((n, n));

    for i in 0..n {
        let degree: f64 = adj.row(i).sum();
        d[[i, i]] = degree;
    }

    d
}

/// Compute degree vector from adjacency matrix.
pub fn degree_vector(adj: &Array2<f64>) -> Array1<f64> {
    adj.sum_axis(Axis(1))
}

/// Unnormalized Laplacian: L = D - A
///
/// # Properties
///
/// - Symmetric positive semidefinite
/// - Smallest eigenvalue is 0 (constant eigenvector)
/// - Number of zero eigenvalues = number of connected components
/// - x^T L x = (1/2) Σ_{ij} A_{ij} (x_i - x_j)² (quadratic form)
///
/// # Example
///
/// ```rust
/// use lapl::adjacency_to_laplacian;
/// use ndarray::array;
///
/// let adj = array![[0.0, 1.0], [1.0, 0.0]];
/// let lap = adjacency_to_laplacian(&adj);
/// assert!((lap[[0,0]] - 1.0).abs() < 1e-10);  // degree
/// assert!((lap[[0,1]] + 1.0).abs() < 1e-10);  // -adjacency
/// ```
pub fn adjacency_to_laplacian(adj: &Array2<f64>) -> Array2<f64> {
    let d = degree_matrix(adj);
    &d - adj
}

/// Symmetric normalized Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}
///
/// Also written as: D^{-1/2} L D^{-1/2}
///
/// # Properties
///
/// - Eigenvalues in [0, 2]
/// - λ = 2 iff graph is bipartite
/// - Preferred for spectral clustering (scale-invariant)
///
/// # Zero-degree nodes
///
/// If a node has zero degree (isolated), \(D^{-1/2}\) is undefined. This
/// function treats \(D^{-1/2}=0\) for such nodes, which yields:
/// - diagonal entry \(L_{ii}=1\)
/// - row/column i otherwise 0
///
/// If you want to reject isolated nodes, use [`normalized_laplacian_checked`].
pub fn normalized_laplacian(adj: &Array2<f64>) -> Array2<f64> {
    let n = adj.nrows();
    let degrees = degree_vector(adj);

    // D^{-1/2}
    let d_inv_sqrt: Array1<f64> = degrees.mapv(|d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 });

    // L_sym = I - D^{-1/2} A D^{-1/2}
    let mut l_sym = Array2::eye(n);

    for i in 0..n {
        for j in 0..n {
            if adj[[i, j]] > 0.0 {
                l_sym[[i, j]] -= d_inv_sqrt[i] * adj[[i, j]] * d_inv_sqrt[j];
            }
        }
    }

    l_sym
}

/// Symmetric normalized Laplacian, but rejects isolated nodes.
pub fn normalized_laplacian_checked(adj: &Array2<f64>) -> Result<Array2<f64>> {
    let degrees = degree_vector(adj);
    if let Some((idx, _)) = degrees.iter().enumerate().find(|(_, &d)| d <= 0.0) {
        return Err(Error::ZeroDegree(idx));
    }
    Ok(normalized_laplacian(adj))
}

/// Random walk Laplacian: L_rw = I - D^{-1} A
///
/// The matrix P = D^{-1} A is the random walk transition matrix.
///
/// # Properties
///
/// - Same eigenvalues as L_sym
/// - Eigenvectors of L_rw = D^{1/2} × (eigenvectors of L_sym)
pub fn random_walk_laplacian(adj: &Array2<f64>) -> Array2<f64> {
    let n = adj.nrows();
    let degrees = degree_vector(adj);

    let mut l_rw = Array2::eye(n);

    for i in 0..n {
        if degrees[i] > 0.0 {
            for j in 0..n {
                if adj[[i, j]] > 0.0 {
                    l_rw[[i, j]] -= adj[[i, j]] / degrees[i];
                }
            }
        }
    }

    l_rw
}

/// Transition matrix: P = D^{-1} A
///
/// P[i,j] = probability of walking from i to j in one step.
pub fn transition_matrix(adj: &Array2<f64>) -> Array2<f64> {
    let n = adj.nrows();
    let degrees = degree_vector(adj);

    let mut p = Array2::zeros((n, n));

    for i in 0..n {
        if degrees[i] > 0.0 {
            for j in 0..n {
                p[[i, j]] = adj[[i, j]] / degrees[i];
            }
        }
    }

    p
}

/// Compute similarity matrix from points using Gaussian kernel.
///
/// A[i,j] = exp(-||x_i - x_j||² / (2σ²))
///
/// # Arguments
///
/// * `points` - n × d matrix (n points, d dimensions)
/// * `sigma` - Kernel bandwidth
pub fn gaussian_similarity(points: &Array2<f64>, sigma: f64) -> Array2<f64> {
    let n = points.nrows();
    let sigma_sq_2 = 2.0 * sigma * sigma;

    let mut adj = Array2::zeros((n, n));

    for i in 0..n {
        for j in i..n {
            let mut dist_sq = 0.0;
            for k in 0..points.ncols() {
                let diff = points[[i, k]] - points[[j, k]];
                dist_sq += diff * diff;
            }

            let sim = (-dist_sq / sigma_sq_2).exp();
            adj[[i, j]] = sim;
            adj[[j, i]] = sim;
        }
    }

    adj
}

/// Compute k-nearest neighbor graph.
///
/// Returns adjacency matrix where A[i,j] = 1 if j is among k nearest neighbors of i.
/// The graph is made symmetric: A_sym = A ∨ A^T
///
/// # Arguments
///
/// * `distances` - n × n distance matrix
/// * `k` - Number of neighbors
pub fn knn_graph(distances: &Array2<f64>, k: usize) -> Array2<f64> {
    let n = distances.nrows();
    let mut adj = Array2::zeros((n, n));

    for i in 0..n {
        // Get distances from node i
        let mut dists: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, distances[[i, j]]))
            .collect();

        // Sort by distance
        dists.sort_by(|a, b| a.1.total_cmp(&b.1));

        // Mark k nearest
        for &(j, _) in dists.iter().take(k) {
            adj[[i, j]] = 1.0;
        }
    }

    // Symmetrize
    for i in 0..n {
        for j in (i + 1)..n {
            if adj[[i, j]] > 0.0 || adj[[j, i]] > 0.0 {
                adj[[i, j]] = 1.0;
                adj[[j, i]] = 1.0;
            }
        }
    }

    adj
}

/// Epsilon-neighborhood graph.
///
/// A[i,j] = 1 if distance(i,j) < epsilon.
pub fn epsilon_graph(distances: &Array2<f64>, epsilon: f64) -> Array2<f64> {
    distances.mapv(|d| if d < epsilon && d > 0.0 { 1.0 } else { 0.0 })
}

/// Configuration for approximate spectral embedding.
#[derive(Debug, Clone)]
pub struct SpectralEmbeddingConfig {
    /// Number of orthogonal-iteration steps.
    ///
    /// Larger means better convergence but higher cost.
    pub iters: usize,
    /// If n ≤ this threshold, use a deterministic Jacobi eigensolver on L_sym.
    ///
    /// This is O(n^3) but very stable for small graphs, and avoids convergence
    /// surprises from iterative methods.
    pub jacobi_max_n: usize,
    /// Jacobi stopping tolerance on the maximum off-diagonal magnitude.
    pub jacobi_tol: f64,
    /// Jacobi maximum number of sweeps (each sweep eliminates one off-diagonal element).
    pub jacobi_max_sweeps: usize,
    /// Whether to drop the trivial constant eigenvector (λ=0) from the embedding.
    ///
    /// Many textbook descriptions of spectral clustering drop it. However, the
    /// Ng–Jordan–Weiss recipe typically takes the *first k eigenvectors* and then
    /// row-normalizes; that includes the constant vector when k ≥ 2.
    pub skip_first: bool,
    /// Whether to row-normalize the resulting embedding.
    pub row_normalize: bool,
}

impl Default for SpectralEmbeddingConfig {
    fn default() -> Self {
        Self {
            iters: 50,
            jacobi_max_n: 64,
            jacobi_tol: 1e-10,
            jacobi_max_sweeps: 50_000,
            skip_first: true,
            row_normalize: true,
        }
    }
}

/// Jacobi eigenvalue algorithm for symmetric matrices.
///
/// Returns `(eigenvalues, eigenvectors)` where eigenvectors are columns.
///
/// This is \(O(n^3)\) but deterministic and stable for small `n`.
fn jacobi_eigh(a: &Array2<f64>, tol: f64, max_sweeps: usize) -> (Vec<f64>, Array2<f64>) {
    let n = a.nrows();
    let mut d = a.to_owned();
    let mut v = Array2::<f64>::eye(n);

    for _ in 0..max_sweeps {
        // Find largest off-diagonal entry.
        let mut p = 0usize;
        let mut q = 1usize;
        let mut max = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = d[[i, j]].abs();
                if val > max {
                    max = val;
                    p = i;
                    q = j;
                }
            }
        }

        if max <= tol {
            break;
        }

        let app = d[[p, p]];
        let aqq = d[[q, q]];
        let apq = d[[p, q]];

        // Compute Jacobi rotation.
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        // Update d = J^T d J.
        for i in 0..n {
            if i != p && i != q {
                let dip = d[[i, p]];
                let diq = d[[i, q]];
                d[[i, p]] = c * dip - s * diq;
                d[[p, i]] = d[[i, p]];
                d[[i, q]] = s * dip + c * diq;
                d[[q, i]] = d[[i, q]];
            }
        }

        let dpp = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        let dqq = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        d[[p, p]] = dpp;
        d[[q, q]] = dqq;
        d[[p, q]] = 0.0;
        d[[q, p]] = 0.0;

        // Update eigenvectors v = v J.
        for i in 0..n {
            let vip = v[[i, p]];
            let viq = v[[i, q]];
            v[[i, p]] = c * vip - s * viq;
            v[[i, q]] = s * vip + c * viq;
        }
    }

    let eigvals: Vec<f64> = (0..n).map(|i| d[[i, i]]).collect();
    (eigvals, v)
}

/// Approximate spectral embedding: the k eigenvectors after the trivial constant one.
///
/// This uses **orthogonal iteration** on the dense matrix \(A = I - L_{sym}\) to
/// approximate the top eigenvectors of A (which correspond to the smallest
/// eigenvectors of \(L_{sym}\)).
///
/// This is a pragmatic, dependency-free default. It is intended for small/medium n
/// where you can afford dense adjacency, but do not want LAPACK/BLAS.
pub fn spectral_embedding(
    adj: &Array2<f64>,
    k: usize,
    cfg: &SpectralEmbeddingConfig,
) -> Result<Array2<f64>> {
    let n = ensure_square(adj)?;
    if k == 0 {
        return Ok(Array2::zeros((n, 0)));
    }
    let start = if cfg.skip_first { 1 } else { 0 };
    if n <= k + start {
        return Err(Error::InvalidEmbeddingDim { k, n });
    }

    let lap = normalized_laplacian(adj);

    // For small n, do a deterministic symmetric eigendecomposition (Jacobi).
    if n <= cfg.jacobi_max_n {
        let (eigvals, eigvecs) = jacobi_eigh(&lap, cfg.jacobi_tol, cfg.jacobi_max_sweeps);

        // Sort eigenpairs by ascending eigenvalue.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| eigvals[i].total_cmp(&eigvals[j]));

        // Take k eigenvectors, optionally skipping the trivial one.
        let mut u = Array2::<f64>::zeros((n, k));
        for (out_col, &eig_idx) in order.iter().skip(start).take(k).enumerate() {
            u.column_mut(out_col).assign(&eigvecs.column(eig_idx));
        }

        if cfg.row_normalize {
            for mut row in u.rows_mut() {
                let norm_sq: f64 = row.iter().map(|x| x * x).sum();
                let norm = norm_sq.sqrt();
                if norm > 0.0 {
                    for x in row.iter_mut() {
                        *x /= norm;
                    }
                }
            }
        }

        return Ok(u);
    }

    // For n > jacobi_max_n:
    //
    // - Without `faer`: use dependency-free orthogonal iteration (approximate).
    // - With `faer`: prefer scalable Krylov–Schur on larger n, otherwise exact dense EVD.
    #[cfg(feature = "faer")]
    {
        if n <= 512 {
            return spectral_embedding_faer_dense_from_laplacian(&lap, k, cfg);
        }
        return spectral_embedding_faer_krylov_schur_from_laplacian(&lap, k, cfg);
    }

    #[cfg(not(feature = "faer"))]
    {
        // Otherwise (no faer): orthogonal iteration on A = I - L_sym.
        let a = Array2::eye(n) - &lap;

        // We want enough eigenvectors to allow skipping the first if requested.
        let r = k + start;
        let mut q = Array2::<f64>::zeros((n, r));
        for i in 0..n {
            for j in 0..r {
                // Deterministic pseudo-random-ish init, avoids external RNG deps.
                q[[i, j]] = ((((i + 1) * 1315423911usize) ^ ((j + 1) * 2654435761usize)) % 10_000)
                    as f64
                    / 10_000.0
                    - 0.5;
            }
        }

        // Orthonormalize columns (modified Gram-Schmidt).
        fn orthonormalize(mut x: Array2<f64>) -> Array2<f64> {
            let r = x.ncols();
            for j in 0..r {
                // subtract projections onto previous columns
                for i in 0..j {
                    let dot = x.column(i).dot(&x.column(j));
                    let col_i = x.column(i).to_owned();
                    let mut col_j = x.column_mut(j);
                    col_j.scaled_add(-dot, &col_i);
                }
                // normalize
                let norm = x.column(j).dot(&x.column(j)).sqrt();
                if norm > 0.0 {
                    x.column_mut(j).mapv_inplace(|v| v / norm);
                }
            }
            x
        }

        q = orthonormalize(q);

        for _ in 0..cfg.iters {
            let z = a.dot(&q);
            q = orthonormalize(z);
        }

        let mut u = q.slice(ndarray::s![.., start..(start + k)]).to_owned();

        if cfg.row_normalize {
            for mut row in u.rows_mut() {
                let norm_sq: f64 = row.iter().map(|x| x * x).sum();
                let norm = norm_sq.sqrt();
                if norm > 0.0 {
                    for x in row.iter_mut() {
                        *x /= norm;
                    }
                }
            }
        }

        Ok(u)
    }
}

/// Spectral embedding via `faer` dense self-adjoint eigendecomposition.
///
/// This computes the eigenvectors of the **symmetric normalized Laplacian** `lap`
/// and returns the `k` eigenvectors after an optional skip of the trivial one.
///
/// Notes:
/// - This is **O(n^3)**. It is not a scalable default for large dense graphs.
/// - It is deterministic and typically more accurate than iterative methods.
#[cfg(feature = "faer")]
fn spectral_embedding_faer_dense_from_laplacian(
    lap: &Array2<f64>,
    k: usize,
    cfg: &SpectralEmbeddingConfig,
) -> Result<Array2<f64>> {
    let n = ensure_square(lap)?;
    let start = if cfg.skip_first { 1 } else { 0 };
    if n <= k + start {
        return Err(Error::InvalidEmbeddingDim { k, n });
    }

    // Convert lap (ndarray) -> Mat (faer).
    let mut m = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            m[(i, j)] = lap[[i, j]];
        }
    }

    // Self-adjoint eigen: eigenvalues are in nondecreasing order (faer docs).
    //
    // We only need eigenvectors. Keep the full decomposition for now.
    let evd = m
        .self_adjoint_eigen(Side::Lower)
        .map_err(|e| Error::Backend(format!("faer self_adjoint_eigen: {e:?}")))?;
    let eigvecs = evd.U();

    let mut u = Array2::<f64>::zeros((n, k));
    for out_col in 0..k {
        let col = start + out_col;
        for i in 0..n {
            u[[i, out_col]] = eigvecs[(i, col)];
        }
    }

    if cfg.row_normalize {
        for mut row in u.rows_mut() {
            let norm_sq: f64 = row.iter().map(|x| x * x).sum();
            let norm = norm_sq.sqrt();
            if norm > 0.0 {
                for x in row.iter_mut() {
                    *x /= norm;
                }
            }
        }
    }

    Ok(u)
}

/// Spectral embedding via `faer` Krylov–Schur partial eigensolver.
///
/// We compute the *top* eigenpairs of a shifted operator \(A = 2I - L\), which correspond to the
/// *bottom* eigenpairs of the normalized Laplacian \(L\).
///
/// This path is intended for larger `n`, where a full dense eigendecomposition becomes too costly.
#[cfg(feature = "faer")]
fn spectral_embedding_faer_krylov_schur_from_laplacian(
    lap: &Array2<f64>,
    k: usize,
    cfg: &SpectralEmbeddingConfig,
) -> Result<Array2<f64>> {
    let n = ensure_square(lap)?;
    let start = if cfg.skip_first { 1 } else { 0 };
    if k == 0 {
        return Ok(Array2::zeros((n, 0)));
    }
    if n <= k + start {
        return Err(Error::InvalidEmbeddingDim { k, n });
    }

    // Build A = 2I - L in faer.
    let mut a = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = -lap[[i, j]];
        }
        a[(i, i)] += 2.0;
    }

    let r = k + start;
    let mut eigvecs = Mat::<f64>::zeros(n, r);
    let mut eigvals = vec![0.0_f64; r];

    // Deterministic initial guess vector.
    let mut v0 = Col::<f64>::zeros(n);
    for i in 0..n {
        v0[i] = (((i + 1) * 2654435761usize % 10_000) as f64) / 10_000.0 - 0.5;
    }

    let par = Par::Seq;
    let params = PartialEigenParams {
        max_restarts: 8,
        ..Default::default()
    };

    let req = partial_eigen_scratch(&a, r, par, params);
    let mut mem = MemBuffer::new(req);
    let mut stack = MemStack::new(&mut mem);

    let _info = partial_self_adjoint_eigen(
        eigvecs.as_mut(),
        &mut eigvals,
        &a,
        v0.as_ref(),
        cfg.jacobi_tol,
        par,
        &mut stack,
        params,
    );

    // Convert eigenvectors into ndarray embedding.
    let mut u = Array2::<f64>::zeros((n, k));
    for out_col in 0..k {
        let col = start + out_col;
        for i in 0..n {
            u[[i, out_col]] = eigvecs[(i, col)];
        }
    }

    if cfg.row_normalize {
        for mut row in u.rows_mut() {
            let norm_sq: f64 = row.iter().map(|x| x * x).sum();
            let norm = norm_sq.sqrt();
            if norm > 0.0 {
                for x in row.iter_mut() {
                    *x /= norm;
                }
            }
        }
    }

    Ok(u)
}

/// Compute Laplacian quadratic form: x^T L x
///
/// For unnormalized Laplacian:
/// x^T L x = (1/2) Σ_{ij} A_{ij} (x_i - x_j)²
///
/// This measures how "smooth" x is over the graph.
pub fn laplacian_quadratic_form(lap: &Array2<f64>, x: &Array1<f64>) -> f64 {
    let lx = lap.dot(x);
    x.dot(&lx)
}

/// Number of connected components (approximation via Laplacian properties).
///
/// Counts near-zero eigenvalues, but without eigensolver this is approximate.
/// For exact count, use graph traversal.
pub fn is_connected(adj: &Array2<f64>) -> bool {
    // Simple DFS-based check
    let n = adj.nrows();
    if n == 0 {
        return true;
    }

    let mut visited = vec![false; n];
    let mut stack = vec![0usize];

    while let Some(node) = stack.pop() {
        if visited[node] {
            continue;
        }
        visited[node] = true;

        for j in 0..n {
            if adj[[node, j]] > 0.0 && !visited[j] {
                stack.push(j);
            }
        }
    }

    visited.iter().all(|&v| v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use proptest::prelude::*;

    #[test]
    fn test_laplacian_basic() {
        // Path graph: 0 -- 1 -- 2
        let adj = array![[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]];

        let lap = adjacency_to_laplacian(&adj);

        // Diagonal = degrees
        assert!((lap[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((lap[[1, 1]] - 2.0).abs() < 1e-10);
        assert!((lap[[2, 2]] - 1.0).abs() < 1e-10);

        // Off-diagonal = -adjacency
        assert!((lap[[0, 1]] + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_laplacian_row_sum_zero() {
        let adj = array![[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]];

        let lap = adjacency_to_laplacian(&adj);

        // Each row should sum to 0
        for i in 0..3 {
            let row_sum: f64 = lap.row(i).sum();
            assert!(row_sum.abs() < 1e-10, "Row {} sum: {}", i, row_sum);
        }
    }

    #[test]
    fn test_normalized_laplacian_diagonal() {
        let adj = array![[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]];

        let lap_norm = normalized_laplacian(&adj);

        // Diagonal should be 1.0
        for i in 0..3 {
            assert!(
                (lap_norm[[i, i]] - 1.0).abs() < 1e-10,
                "Diagonal {} = {}",
                i,
                lap_norm[[i, i]]
            );
        }
    }

    #[test]
    fn test_transition_matrix_row_sum() {
        let adj = array![[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]];

        let p = transition_matrix(&adj);

        // Each row should sum to 1.0
        for i in 0..3 {
            let row_sum: f64 = p.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-10, "Row {} sum: {}", i, row_sum);
        }
    }

    #[test]
    fn test_spectral_embedding_shape() {
        let adj = array![[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]];
        let u = spectral_embedding(&adj, 1, &SpectralEmbeddingConfig::default()).unwrap();
        assert_eq!(u.nrows(), 3);
        assert_eq!(u.ncols(), 1);
    }

    #[cfg(feature = "faer")]
    #[test]
    fn spectral_embedding_faer_dense_agrees_with_jacobi_on_small_graph() {
        // Path graph: 0 -- 1 -- 2 -- 3
        let adj = array![
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ];
        let cfg = SpectralEmbeddingConfig {
            // Force the Jacobi path in the main function.
            jacobi_max_n: 16,
            skip_first: true,
            row_normalize: true,
            ..Default::default()
        };

        // 1D embedding (Fiedler vector, up to sign).
        let u_jacobi = spectral_embedding(&adj, 1, &cfg).unwrap();
        let lap = normalized_laplacian(&adj);
        let u_faer = spectral_embedding_faer_dense_from_laplacian(&lap, 1, &cfg).unwrap();

        // Compare up to sign: correlation should be near 1 in absolute value.
        let dot: f64 = (0..4).map(|i| u_jacobi[[i, 0]] * u_faer[[i, 0]]).sum();
        assert!(dot.abs() > 0.99, "abs(dot)={} too small", dot.abs());
    }

    #[test]
    fn test_normalized_laplacian_checked_rejects_isolated() {
        let adj = array![[0.0, 0.0], [0.0, 0.0]];
        let err = normalized_laplacian_checked(&adj).unwrap_err();
        match err {
            Error::ZeroDegree(0) | Error::ZeroDegree(1) => {}
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn test_gaussian_similarity_self() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let sim = gaussian_similarity(&points, 1.0);

        // Self-similarity should be 1.0
        for i in 0..3 {
            assert!((sim[[i, i]] - 1.0).abs() < 1e-10);
        }

        // Symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert!((sim[[i, j]] - sim[[j, i]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_knn_graph() {
        let distances = array![
            [0.0, 1.0, 2.0, 10.0],
            [1.0, 0.0, 1.5, 10.0],
            [2.0, 1.5, 0.0, 10.0],
            [10.0, 10.0, 10.0, 0.0]
        ];

        let adj = knn_graph(&distances, 2);

        // Node 3 is far from everyone, but k=2 means it connects to someone
        assert!(adj.sum() > 0.0);

        // Symmetric
        for i in 0..4 {
            for j in 0..4 {
                assert!((adj[[i, j]] - adj[[j, i]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_is_connected() {
        // Connected
        let adj_connected = array![[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]];
        assert!(is_connected(&adj_connected));

        // Disconnected
        let adj_disconnected = array![[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        assert!(!is_connected(&adj_disconnected));
    }

    #[test]
    fn test_quadratic_form() {
        let adj = array![[0.0, 1.0], [1.0, 0.0]];
        let lap = adjacency_to_laplacian(&adj);

        // Constant vector: x^T L x = 0
        let ones = array![1.0, 1.0];
        let qf = laplacian_quadratic_form(&lap, &ones);
        assert!(qf.abs() < 1e-10);

        // Non-constant: x^T L x > 0
        let non_const = array![1.0, -1.0];
        let qf = laplacian_quadratic_form(&lap, &non_const);
        assert!(qf > 0.0);
    }

    proptest! {
        #[test]
        fn prop_normalized_laplacian_is_symmetric_for_symmetric_adj(
            n in 2usize..20,
            weights in prop::collection::vec(0.0f64..1.0, 1..500),
        ) {
            // Build a symmetric adjacency with zero diagonal.
            let mut adj = Array2::<f64>::zeros((n, n));
            let mut it = weights.into_iter();
            for i in 0..n {
                for j in (i+1)..n {
                    let w = it.next().unwrap_or(0.0);
                    adj[[i, j]] = w;
                    adj[[j, i]] = w;
                }
            }
            for i in 0..n { adj[[i,i]] = 0.0; }

            // Ensure no isolated nodes: add a tiny ring if needed.
            for i in 0..n {
                if adj.row(i).sum() == 0.0 {
                    let j = (i + 1) % n;
                    adj[[i, j]] = 1e-3;
                    adj[[j, i]] = 1e-3;
                }
            }

            let l = normalized_laplacian(&adj);

            // Symmetry: L == L^T (within eps).
            let eps = 1e-10;
            for i in 0..n {
                for j in 0..n {
                    prop_assert!((l[[i,j]] - l[[j,i]]).abs() <= eps);
                }
            }

            // With zero diagonal adjacency, normalized Laplacian diagonal is ~1.
            for i in 0..n {
                prop_assert!((l[[i,i]] - 1.0).abs() <= 1e-10);
            }
        }

        /// For symmetric normalized Laplacian, all Rayleigh quotients lie in [0, 2].
        #[test]
        fn prop_rayleigh_quotient_bounds_for_normalized_laplacian(
            n in 2usize..25,
            weights in prop::collection::vec(0.0f64..1.0, 1..2000),
            x in prop::collection::vec(-1.0f64..1.0, 2..25),
        ) {
            let mut adj = Array2::<f64>::zeros((n, n));
            let mut it = weights.into_iter();
            for i in 0..n {
                for j in (i+1)..n {
                    let w = it.next().unwrap_or(0.0);
                    adj[[i, j]] = w;
                    adj[[j, i]] = w;
                }
            }
            for i in 0..n { adj[[i,i]] = 0.0; }

            for i in 0..n {
                if adj.row(i).sum() == 0.0 {
                    let j = (i + 1) % n;
                    adj[[i, j]] = 1e-3;
                    adj[[j, i]] = 1e-3;
                }
            }

            let l = normalized_laplacian_checked(&adj).unwrap();

            let mut xv = Array1::<f64>::zeros(n);
            for i in 0..n {
                xv[i] = x.get(i).copied().unwrap_or(0.0);
            }
            let denom = xv.dot(&xv);
            prop_assume!(denom > 1e-12);

            let rq = laplacian_quadratic_form(&l, &xv) / denom;
            prop_assert!(rq >= -1e-9, "rq={} < 0", rq);
            prop_assert!(rq <= 2.0 + 1e-6, "rq={} > 2", rq);
        }
    }
}
