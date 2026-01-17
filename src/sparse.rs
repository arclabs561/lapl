//! Sparse graph utilities for spectral methods.
//!
//! This module is intentionally small:
//! - a minimal CSR adjacency representation (undirected, weighted)
//! - a matrix-free spectral embedding via orthogonal iteration on
//!   \(S = D^{-1/2} A D^{-1/2}\), avoiding dense \(n \times n\) matrices.
//!
//! The intent is to unblock scaling beyond dense adjacency without committing to
//! a full sparse linear algebra dependency stack.

use crate::{Error, Result, SpectralEmbeddingConfig};
use ndarray::{Array2, ArrayView2};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// A minimal CSR adjacency matrix for an undirected weighted graph.
///
/// Stores edges as rows with `(col, weight)` pairs.
#[derive(Debug, Clone)]
pub struct CsrAdjacency {
    n: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    values: Vec<f64>,
}

impl CsrAdjacency {
    /// Build an undirected adjacency from an edge list.
    ///
    /// Each edge `(u, v, w)` inserts both `(u, v)` and `(v, u)` with weight `w`.
    /// If multiple edges target the same `(row, col)`, we keep the **maximum**
    /// weight (common for “similarity” graphs).
    pub fn from_undirected_edges(n: usize, edges: &[(usize, usize, f64)]) -> Result<Self> {
        if n == 0 {
            return Ok(Self {
                n,
                row_ptr: vec![0],
                col_idx: vec![],
                values: vec![],
            });
        }

        let mut entries: Vec<(usize, usize, f64)> = Vec::with_capacity(edges.len() * 2);
        for &(u, v, w) in edges {
            if u >= n || v >= n {
                return Err(Error::Backend(format!(
                    "edge index out of bounds: u={}, v={}, n={}",
                    u, v, n
                )));
            }
            if u == v {
                // Skip self-loops: most spectral formulations assume zero diagonal.
                continue;
            }
            entries.push((u, v, w));
            entries.push((v, u, w));
        }

        // Sort by (row, col).
        entries.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));

        // Deduplicate by keeping the maximum weight per (row, col).
        let mut dedup: Vec<(usize, usize, f64)> = Vec::with_capacity(entries.len());
        for (r, c, w) in entries {
            if let Some((pr, pc, pw)) = dedup.last_mut() {
                if *pr == r && *pc == c {
                    if w > *pw {
                        *pw = w;
                    }
                    continue;
                }
            }
            dedup.push((r, c, w));
        }

        let mut row_ptr = vec![0usize; n + 1];
        for &(r, _, _) in &dedup {
            row_ptr[r + 1] += 1;
        }
        for i in 0..n {
            row_ptr[i + 1] += row_ptr[i];
        }

        let nnz = dedup.len();
        let mut col_idx = vec![0usize; nnz];
        let mut values = vec![0.0f64; nnz];

        // Write entries into CSR using row_ptr as offsets.
        let mut offsets = row_ptr.clone();
        for (r, c, w) in dedup {
            let at = offsets[r];
            col_idx[at] = c;
            values[at] = w;
            offsets[r] += 1;
        }

        Ok(Self {
            n,
            row_ptr,
            col_idx,
            values,
        })
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// Row sums (degrees) for this adjacency.
    pub fn degree_vector(&self) -> Vec<f64> {
        #[cfg(feature = "parallel")]
        {
            (0..self.n)
                .into_par_iter()
                .map(|r| {
                    let start = self.row_ptr[r];
                    let end = self.row_ptr[r + 1];
                    let mut sum = 0.0;
                    for idx in start..end {
                        sum += self.values[idx];
                    }
                    sum
                })
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            let mut deg = vec![0.0f64; self.n];
            for (r, deg_r) in deg.iter_mut().enumerate() {
                let start = self.row_ptr[r];
                let end = self.row_ptr[r + 1];
                let mut sum = 0.0;
                for idx in start..end {
                    sum += self.values[idx];
                }
                *deg_r = sum;
            }
            deg
        }
    }
}

fn orthonormalize(mut x: Array2<f64>) -> Array2<f64> {
    let r = x.ncols();
    for j in 0..r {
        for i in 0..j {
            let dot = x.column(i).dot(&x.column(j));
            let col_i = x.column(i).to_owned();
            let mut col_j = x.column_mut(j);
            col_j.scaled_add(-dot, &col_i);
        }
        let norm = x.column(j).dot(&x.column(j)).sqrt();
        if norm > 0.0 {
            x.column_mut(j).mapv_inplace(|v| v / norm);
        }
    }
    x
}

fn apply_normalized_similarity_batch(
    adj: &CsrAdjacency,
    d_inv_sqrt: &[f64],
    q: ArrayView2<'_, f64>,
) -> Array2<f64> {
    let n = adj.n;
    let r_cols = q.ncols();
    let mut y = Array2::<f64>::zeros((n, r_cols));

    #[cfg(feature = "parallel")]
    {
        use ndarray::parallel::prelude::*;
        y.axis_iter_mut(ndarray::Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(r, mut y_row)| {
                let dr = d_inv_sqrt[r];
                if dr > 0.0 {
                    let start = adj.row_ptr[r];
                    let end = adj.row_ptr[r + 1];
                    for idx in start..end {
                        let c = adj.col_idx[idx];
                        let w = adj.values[idx];
                        let dc = d_inv_sqrt[c];
                        let w_norm = dr * w * dc;
                        let q_row = q.row(c);
                        for j in 0..r_cols {
                            y_row[j] += w_norm * q_row[j];
                        }
                    }
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for r in 0..n {
            let dr = d_inv_sqrt[r];
            if dr == 0.0 {
                continue;
            }
            let start = adj.row_ptr[r];
            let end = adj.row_ptr[r + 1];
            for idx in start..end {
                let c = adj.col_idx[idx];
                let w = adj.values[idx];
                let dc = d_inv_sqrt[c];
                let w_norm = dr * w * dc;
                for j in 0..r_cols {
                    y[[r, j]] += w_norm * q[[c, j]];
                }
            }
        }
    }

    y
}

/// Matrix-free spectral embedding for a sparse adjacency.
///
/// We avoid forming the dense normalized Laplacian. Instead, we perform
/// orthogonal iteration on \(S = D^{-1/2} A D^{-1/2}\), whose **top**
/// eigenvectors correspond to the **bottom** eigenvectors of
/// \(L_{sym} = I - S\).
pub fn spectral_embedding_sparse(
    adj: &CsrAdjacency,
    k: usize,
    cfg: &SpectralEmbeddingConfig,
) -> Result<Array2<f64>> {
    let n = adj.n();
    if k == 0 {
        return Ok(Array2::zeros((n, 0)));
    }

    // Mirror the dense API’s "skip_first" handling.
    let start = if cfg.skip_first { 1 } else { 0 };
    if n <= k + start {
        return Err(Error::InvalidEmbeddingDim { k, n });
    }

    // Degrees and D^{-1/2}.
    let deg = adj.degree_vector();
    let mut d_inv_sqrt = vec![0.0f64; n];
    for i in 0..n {
        let d = deg[i];
        if d > 0.0 {
            d_inv_sqrt[i] = 1.0 / d.sqrt();
        }
    }

    // We want r = k + start eigenvectors from S, then slice away the first if requested.
    let r = k + start;
    let mut q = Array2::<f64>::zeros((n, r));
    for i in 0..n {
        for j in 0..r {
            // Deterministic init, same spirit as the dense orthogonal-iteration path.
            q[[i, j]] = ((((i + 1) * 1315423911usize) ^ ((j + 1) * 2654435761usize)) % 10_000)
                as f64
                / 10_000.0
                - 0.5;
        }
    }
    q = orthonormalize(q);

    for _ in 0..cfg.iters {
        // Z = S * Q, computed by sparse mat-matrix multiply.
        let z = apply_normalized_similarity_batch(adj, &d_inv_sqrt, q.view());
        q = orthonormalize(z);
    }

    let mut u = q.slice(ndarray::s![.., start..(start + k)]).to_owned();

    if cfg.row_normalize {
        #[cfg(feature = "parallel")]
        {
            use ndarray::parallel::prelude::*;
            u.axis_iter_mut(ndarray::Axis(0))
                .into_par_iter()
                .for_each(|mut row| {
                    let norm_sq: f64 = row.iter().map(|x| x * x).sum();
                    let norm = norm_sq.sqrt();
                    if norm > 0.0 {
                        row.mapv_inplace(|x| x / norm);
                    }
                });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for mut row in u.rows_mut() {
                let norm_sq: f64 = row.iter().map(|x| x * x).sum();
                let norm = norm_sq.sqrt();
                if norm > 0.0 {
                    row.mapv_inplace(|x| x / norm);
                }
            }
        }
    }

    Ok(u)
}
