# lapl

[![crates.io](https://img.shields.io/crates/v/lapl.svg)](https://crates.io/crates/lapl)
[![Documentation](https://docs.rs/lapl/badge.svg)](https://docs.rs/lapl)
[![CI](https://github.com/arclabs561/lapl/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/lapl/actions/workflows/ci.yml)

Spectral graph methods.

Dual-licensed under MIT or Apache-2.0.

[crates.io](https://crates.io/crates/lapl) | [docs.rs](https://docs.rs/lapl)

## Quickstart

```toml
[dependencies]
lapl = "0.1.0"
ndarray = "0.16"
```

```rust
use lapl::{adjacency_to_laplacian, normalized_laplacian, gaussian_similarity};
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

## The Laplacian Zoo

- **Unnormalized L = D - A**: Simple but scale-dependent
- **Normalized L_sym**: Eigenvalues in [0, 2], used for spectral clustering
- **Random walk L_rw**: Same spectrum as L_sym, different eigenvectors
