# lapl

[![crates.io](https://img.shields.io/crates/v/lapl.svg)](https://crates.io/crates/lapl)
[![Documentation](https://docs.rs/lapl/badge.svg)](https://docs.rs/lapl)

Spectral graph methods.

See [examples/README.md](examples/README.md) for the runnable spectral
diagnostic example.

## Quickstart

```toml
[dependencies]
lapl = "0.2"
ndarray = "0.16"
```

```rust
use lapl::{adjacency_to_laplacian, normalized_laplacian};
use ndarray::array;

// Simple graph: 0 -- 1 -- 2
let adj = array![
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0]
];

let lap = adjacency_to_laplacian(&adj);       // L = D - A
let lap_norm = normalized_laplacian(&adj);    // L_sym = I - D^{-1/2} A D^{-1/2}
```

## Functions

| Function | Purpose |
|----------|---------|
| `adjacency_to_laplacian` | Unnormalized L = D - A |
| `normalized_laplacian` | Symmetric normalized (diagonal=1 for isolated nodes) |
| `normalized_laplacian_checked` | Rejects graphs with isolated nodes |
| `random_walk_laplacian` | L_rw = I - D^{-1} A |
| `transition_matrix` | Random walk P = D^{-1} A |
| `gaussian_similarity` | RBF kernel similarity |
| `knn_graph` | k-nearest neighbor graph |
| `epsilon_graph` | Epsilon-neighborhood |
| `is_connected` | Check connectivity |
| `laplacian_quadratic_form` | x^T L x |
| `symmetric_eigenvalues` | Eigenvalues for symmetric matrices |

## Limits

- Everything is dense: adjacency matrices and Laplacians are `n x n`
  `ndarray` arrays, so memory is `O(n^2)`.
- `symmetric_eigenvalues` and the small-graph path of `spectral_embedding`
  (`n <= jacobi_max_n`, default 64) use classical Jacobi rotations. Each
  rotation scans all off-diagonal entries and convergence takes on the order
  of `n^2` rotations, so cost grows roughly as `n^4`. The `max_sweeps` cap
  counts rotations; when it is reached the result is returned unconverged.
- Larger graphs in `spectral_embedding` use orthogonal iteration (approximate),
  or with the `faer` feature a dense or Krylov-Schur eigensolver. The `sparse`
  feature adds a matrix-free embedding (`sparse::spectral_embedding_sparse`)
  over a CSR adjacency.

## The Laplacian Zoo

- **Unnormalized L = D - A**: Simple but scale-dependent
- **Normalized L_sym**: Eigenvalues in [0, 2], used for spectral clustering
- **Random walk L_rw**: Same spectrum as L_sym, different eigenvectors

## License

Licensed under either the [Apache License, Version 2.0](LICENSE-APACHE) or
the [MIT license](LICENSE-MIT), at your option.
