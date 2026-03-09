//! Spectral analysis of graph Laplacians using Random Matrix Theory.
//!
//! Computes the Laplacian of random and structured graphs, extracts eigenvalues,
//! and uses rmt to analyze whether the spectrum looks random (Wigner-like) or
//! has structure (community signal above the MP/Wigner bulk).
//!
//! Key idea: eigenvalues below the RMT noise floor are noise; those above are
//! signal revealing genuine graph structure (communities, hubs, bottlenecks).
//!
//! Run: cargo run --example spectral_rmt

use lapl::{adjacency_to_laplacian, gaussian_similarity, normalized_laplacian};
use ndarray::Array2;
use rmt::{empirical_spectral_density, level_spacing_ratios, mean_spacing_ratio};

fn main() {
    println!("=== Graph Laplacian Spectral Analysis via RMT ===\n");

    // 1. Random graph (Erdos-Renyi style via Gaussian similarity)
    let n = 60;
    let random_points = random_points_2d(n, 42);
    let adj_random = gaussian_similarity(&random_points, 1.0);
    let lap_random = normalized_laplacian(&adj_random);
    let eigs_random = eigenvalues_symmetric(&lap_random);

    // 2. Structured graph (3 well-separated clusters)
    let structured_points = clustered_points_2d(20, 3, 5.0, 42);
    let adj_struct = gaussian_similarity(&structured_points, 1.0);
    let lap_struct = normalized_laplacian(&adj_struct);
    let eigs_struct = eigenvalues_symmetric(&lap_struct);

    // Analyze with RMT
    println!("--- Random graph ({n} nodes, Gaussian similarity) ---\n");
    analyze_spectrum("Random", &eigs_random);

    println!("\n--- Structured graph (3 clusters of 20, separation=5.0) ---\n");
    analyze_spectrum("Structured", &eigs_struct);

    // Compare unnormalized Laplacian
    let lap_unnorm = adjacency_to_laplacian(&adj_struct);
    let eigs_unnorm = eigenvalues_symmetric(&lap_unnorm);
    println!("\n--- Structured (unnormalized Laplacian) ---\n");
    analyze_spectrum("Unnormalized", &eigs_unnorm);
}

fn analyze_spectrum(label: &str, eigenvalues: &[f64]) {
    let n = eigenvalues.len();

    // Spacing ratio: GOE ~ 0.5307, Poisson ~ 0.3863
    let msr = mean_spacing_ratio(eigenvalues);
    let ratios = level_spacing_ratios(eigenvalues);
    let regime = if msr > 0.48 {
        "GOE (correlated/repulsive)"
    } else if msr < 0.42 {
        "Poisson (uncorrelated/independent)"
    } else {
        "intermediate"
    };

    println!("  {label}: {n} eigenvalues");
    println!(
        "  Range: [{:.4}, {:.4}]",
        eigenvalues[0],
        eigenvalues[n - 1]
    );
    println!("  Mean spacing ratio: {msr:.4} -> {regime}");

    // Count near-zero eigenvalues (connected components indicator)
    let near_zero = eigenvalues.iter().filter(|&&e| e.abs() < 1e-6).count();
    println!("  Near-zero eigenvalues: {near_zero} (= connected components)");

    // Effective dimension via RMT
    let effective = rmt::effective_dimension(eigenvalues, n, n);
    println!("  Signal dimensions (above MP bulk): {effective}");

    // Histogram summary
    let (centers, densities) = empirical_spectral_density(eigenvalues, 10);
    println!("  Spectral density (10 bins):");
    for (c, d) in centers.iter().zip(densities.iter()) {
        let bar_len = (d * 20.0) as usize;
        let bar = "#".repeat(bar_len.min(40));
        println!("    {c:6.3} | {d:5.2} {bar}");
    }

    // Spacing ratio distribution summary
    if ratios.len() >= 5 {
        let mut sorted_ratios = ratios.clone();
        sorted_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q1 = sorted_ratios[sorted_ratios.len() / 4];
        let median = sorted_ratios[sorted_ratios.len() / 2];
        let q3 = sorted_ratios[3 * sorted_ratios.len() / 4];
        println!("  Spacing ratio quartiles: Q1={q1:.3} med={median:.3} Q3={q3:.3}");
    }
}

/// Simple eigenvalue computation via Jacobi iteration (for small symmetric matrices).
fn eigenvalues_symmetric(m: &Array2<f64>) -> Vec<f64> {
    let n = m.nrows();
    let mut a = m.clone();
    // Jacobi eigenvalue algorithm for symmetric matrices
    for _ in 0..100 * n * n {
        // Find largest off-diagonal element
        let mut max_val = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if a[[i, j]].abs() > max_val {
                    max_val = a[[i, j]].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-12 {
            break;
        }
        // Compute rotation
        let theta = if (a[[p, p]] - a[[q, q]]).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * a[[p, q]] / (a[[p, p]] - a[[q, q]])).atan()
        };
        let c = theta.cos();
        let s = theta.sin();
        // Apply rotation
        let mut new_a = a.clone();
        for i in 0..n {
            if i != p && i != q {
                new_a[[i, p]] = c * a[[i, p]] + s * a[[i, q]];
                new_a[[p, i]] = new_a[[i, p]];
                new_a[[i, q]] = -s * a[[i, p]] + c * a[[i, q]];
                new_a[[q, i]] = new_a[[i, q]];
            }
        }
        new_a[[p, p]] = c * c * a[[p, p]] + 2.0 * s * c * a[[p, q]] + s * s * a[[q, q]];
        new_a[[q, q]] = s * s * a[[p, p]] - 2.0 * s * c * a[[p, q]] + c * c * a[[q, q]];
        new_a[[p, q]] = 0.0;
        new_a[[q, p]] = 0.0;
        a = new_a;
    }
    let mut eigs: Vec<f64> = (0..n).map(|i| a[[i, i]]).collect();
    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eigs
}

fn random_points_2d(n: usize, seed: u64) -> Array2<f64> {
    // Simple LCG for reproducible random points
    let mut state = seed;
    let mut points = Array2::zeros((n, 2));
    for i in 0..n {
        for j in 0..2 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            points[[i, j]] = (state >> 33) as f64 / (1u64 << 31) as f64;
        }
    }
    points
}

fn clustered_points_2d(
    per_cluster: usize,
    n_clusters: usize,
    separation: f64,
    seed: u64,
) -> Array2<f64> {
    let n = per_cluster * n_clusters;
    let mut points = Array2::zeros((n, 2));
    let mut state = seed;
    for c in 0..n_clusters {
        let cx = (c as f64) * separation;
        let cy = 0.0;
        for i in 0..per_cluster {
            let idx = c * per_cluster + i;
            for j in 0..2 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let noise = ((state >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.5;
                points[[idx, j]] = if j == 0 { cx } else { cy } + noise;
            }
        }
    }
    points
}
