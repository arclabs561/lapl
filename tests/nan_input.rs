//! Sorting paths (k-NN neighbour order, eigenvalue order) must not panic on NaN.

use lapl::{knn_graph, symmetric_eigenvalues};
use ndarray::array;

#[test]
fn knn_graph_with_nan_distances_does_not_panic() {
    let d = array![
        [0.0, 1.0, f64::NAN, 2.0],
        [1.0, 0.0, 0.5, f64::NAN],
        [f64::NAN, 0.5, 0.0, 1.5],
        [2.0, f64::NAN, 1.5, 0.0]
    ];
    let adj = knn_graph(&d, 2);
    assert_eq!(adj.dim(), (4, 4));
}

#[test]
fn symmetric_eigenvalues_with_nan_entry_does_not_panic() {
    let a = array![[2.0, f64::NAN, 0.0], [f64::NAN, 1.0, 0.5], [0.0, 0.5, 3.0]];
    let _ = symmetric_eigenvalues(&a, 1e-10, 100);
}
