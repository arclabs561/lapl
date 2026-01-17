//! Scaling benchmark for spectral embedding: Dense vs Sparse.

// This bench compares `lapl::spectral_embedding` (dense) vs `lapl::sparse::spectral_embedding_sparse`.
//
// `lapl::sparse` is behind the `sparse` feature, but `cargo clippy --all-targets`
// builds benches without enabling features by default.
//
// So: compile a no-op bench binary when `sparse` is disabled.

#[cfg(feature = "sparse")]
mod sparse_enabled {
    use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
    use lapl::sparse::{spectral_embedding_sparse, CsrAdjacency};
    use lapl::{spectral_embedding, SpectralEmbeddingConfig};
    use ndarray::Array2;

    fn build_path_graph_dense(n: usize) -> Array2<f64> {
        let mut adj = Array2::zeros((n, n));
        for i in 0..n - 1 {
            adj[[i, i + 1]] = 1.0;
            adj[[i + 1, i]] = 1.0;
        }
        adj
    }

    fn build_path_graph_sparse(n: usize) -> CsrAdjacency {
        let mut edges = Vec::with_capacity(n * 2);
        for i in 0..n - 1 {
            edges.push((i, i + 1, 1.0));
        }
        CsrAdjacency::from_undirected_edges(n, &edges).unwrap()
    }

    fn bench_spectral_scaling(c: &mut Criterion) {
        let mut group = c.benchmark_group("spectral_embedding_scaling");

        // We test a range where dense starts to hit the wall
        for n in [64usize, 128, 256, 512, 1024] {
            let k = 4;
            let cfg = SpectralEmbeddingConfig {
                iters: 20,        // Keep iterations constant for fair comparison
                jacobi_max_n: 64, // Beyond this, dense uses Faer or OrthIter
                ..Default::default()
            };

            let dense_adj = build_path_graph_dense(n);
            group.bench_with_input(BenchmarkId::new("dense", n), &n, |b, _| {
                b.iter(|| black_box(spectral_embedding(black_box(&dense_adj), k, &cfg)).unwrap())
            });

            let sparse_adj = build_path_graph_sparse(n);
            group.bench_with_input(BenchmarkId::new("sparse", n), &n, |b, _| {
                b.iter(|| {
                    black_box(spectral_embedding_sparse(black_box(&sparse_adj), k, &cfg)).unwrap()
                })
            });
        }
        group.finish();
    }

    criterion_group!(benches, bench_spectral_scaling);
    criterion_main!(benches);
}

#[cfg(not(feature = "sparse"))]
fn main() {}
