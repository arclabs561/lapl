//! The multiplicity of the zero eigenvalue of a graph Laplacian equals the
//! number of connected components (von Luxburg 2007, Prop. 2 and 4).

use lapl::{adjacency_to_laplacian, normalized_laplacian, symmetric_eigenvalues};
use ndarray::Array2;

fn adjacency(n: usize, edges: &[(usize, usize)]) -> Array2<f64> {
    let mut a = Array2::zeros((n, n));
    for &(i, j) in edges {
        a[[i, j]] = 1.0;
        a[[j, i]] = 1.0;
    }
    a
}

fn zero_count(eigs: &[f64]) -> usize {
    eigs.iter().filter(|l| l.abs() < 1e-8).count()
}

#[test]
fn unnormalized_zero_multiplicity_counts_components() {
    // Triangle {0,1,2}, edge {3,4}, isolated node 5: three components.
    // Spectra: triangle {0,3,3}, edge {0,2}, isolated {0}.
    let adj = adjacency(6, &[(0, 1), (1, 2), (0, 2), (3, 4)]);
    let eigs = symmetric_eigenvalues(&adjacency_to_laplacian(&adj), 1e-12, 10_000).unwrap();
    assert_eq!(zero_count(&eigs), 3, "eigenvalues {eigs:?}");
    let expected = [0.0, 0.0, 0.0, 2.0, 3.0, 3.0];
    for (got, want) in eigs.iter().zip(expected) {
        assert!((got - want).abs() < 1e-9, "eigenvalues {eigs:?}");
    }
}

#[test]
fn normalized_zero_multiplicity_counts_components_without_isolated_nodes() {
    // A 4-cycle and a triangle: two components, no isolated nodes.
    let adj = adjacency(7, &[(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (4, 6)]);
    let eigs = symmetric_eigenvalues(&normalized_laplacian(&adj), 1e-12, 10_000).unwrap();
    assert_eq!(zero_count(&eigs), 2, "eigenvalues {eigs:?}");
}
