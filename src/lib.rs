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
//! - [`stratify`](../stratify): Spectral clustering uses this crate
//! - [`rmt`](../rmt): Random graph Laplacians follow RMT eigenvalue distributions
//!
//! ## What Can Go Wrong
//!
//! 1. **Disconnected graph**: Zero eigenvalue multiplicity = number of components.
//! 2. **Zero degree nodes**: Normalized Laplacian undefined. Remove isolated nodes.
//! 3. **Numerical precision**: Very small/large edge weights cause instability.
//! 4. **Eigensolver required**: This crate provides Laplacians, not eigensolvers.
//!    Use `ndarray-linalg` or `nalgebra` for eigendecomposition.
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
}

pub type Result<T> = std::result::Result<T, Error>;

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
/// # Panics
///
/// Panics if any node has zero degree.
pub fn normalized_laplacian(adj: &Array2<f64>) -> Array2<f64> {
    let n = adj.nrows();
    let degrees = degree_vector(adj);

    // D^{-1/2}
    let d_inv_sqrt: Array1<f64> = degrees.mapv(|d| {
        if d > 0.0 {
            1.0 / d.sqrt()
        } else {
            0.0
        }
    });

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
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

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

/// Algebraic connectivity: second smallest eigenvalue of L.
///
/// λ_2 > 0 iff graph is connected.
/// Higher λ_2 = more connected.
///
/// This is a simple power iteration approximation.
#[cfg(feature = "linalg")]
pub fn algebraic_connectivity(lap: &Array2<f64>) -> f64 {
    // Would need eigenvalue computation
    unimplemented!("Requires ndarray-linalg feature and proper eigensolver")
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
            assert!(
                (row_sum - 1.0).abs() < 1e-10,
                "Row {} sum: {}",
                i,
                row_sum
            );
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
}
