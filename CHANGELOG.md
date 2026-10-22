# Changelog

## [Unreleased]

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
