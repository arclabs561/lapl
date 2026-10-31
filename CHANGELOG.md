# Changelog

## [Unreleased]

## [0.3.0] - 2026-10-30

### Added

- `spectral_embedding_with_diagnostics` and
  `sparse::spectral_embedding_sparse_with_diagnostics` return the embedding
  together with `EmbeddingDiagnostics`: per-column eigenvalue estimates,
  residuals `||L u - theta u||`, the tolerance and a `converged` flag.

### Fixed

- Dense `spectral_embedding` above `jacobi_max_n` (without `faer`) and every
  `spectral_embedding_sparse` call ran a fixed 50 steps of orthogonal
  iteration with no convergence check, returning eigenvectors far from the
  true ones on graphs with small spectral gaps (subspace angles of 71 to 89
  degrees on a 100-node path). Both now use a block Krylov solver with thick
  restarts that iterates until every residual is within
  `jacobi_tol * sqrt(n)`, capped at `iters` restart cycles (default raised
  from 50 to 300), and return `Error::Backend` when the cap is reached first.
- With `faer`, graphs above `n = 512` used a Krylov-Schur solver started
  from a single vector, which cannot resolve a repeated eigenvalue such as
  the double zero eigenvalue of a graph with two components. Those sizes now
  use the block solver, whose start block has one column per wanted
  eigenvector.

### Documentation

- State the symmetry and finite, nonnegative weight preconditions for dense
  spectral methods, and clarify that `normalized_laplacian_checked` checks
  only row degrees. Runtime behavior is unchanged.
- Correct the documented eigenvector conversion from the symmetric normalized
  Laplacian to the random-walk Laplacian to use `D^{-1/2}`.

## [0.2.0] - 2026-07-05

### Added

- Added `symmetric_eigenvalues` for callers that need eigenvalues without
  constructing a full embedding.

## [0.1.2] - 2026-06-10

### Added

- petgraph adjacency-matrix conversion.
- `spectral_rmt` example connecting graph Laplacians to random matrix theory.
