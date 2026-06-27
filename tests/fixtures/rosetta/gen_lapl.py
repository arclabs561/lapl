# /// script
# requires-python = ">=3.10"
# dependencies = ["scipy", "numpy"]
# ///
"""Rosetta fixture generator for lapl graph Laplacians and spectral embedding.

Provenance for lapl_spectral.json.

Matrix oracles (EXACT, element-wise):
  adjacency_to_laplacian  -> scipy.sparse.csgraph.laplacian(A, normed=False)  [D - A]
  normalized_laplacian    -> scipy.sparse.csgraph.laplacian(A, normed=True)   [I - D^-1/2 A D^-1/2]
  random_walk_laplacian   -> numpy I - D^-1 A  (no scipy function; trivial formula)

Spectral embedding (SUBSPACE class): eigenvectors are unique only up to sign and,
under degeneracy, rotation. So we do NOT compare eigenvectors element-wise.
Instead the fixture ships the reference invariant subspace (the k smallest
eigenvectors of L_sym from numpy.linalg.eigh) and the Rust test checks that each
reference eigenvector lies in the span of lapl's spectral_embedding output
(projection residual ~ 0). A complete weighted graph is used so the eigenvalues
are distinct and the smallest-k subspace is well defined.

Regenerate: uv run tests/fixtures/rosetta/gen_lapl.py
"""

import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.sparse.csgraph import laplacian as sp_laplacian

SEED = 0
rng = np.random.default_rng(SEED)

# Complete weighted undirected graph on 6 nodes: connected, no isolated nodes,
# generically distinct eigenvalues so the smallest-k subspace is unambiguous.
n = 6
k = 2
w = rng.uniform(0.5, 2.0, size=(n, n))
adj = np.triu(w, 1)
adj = adj + adj.T  # symmetric, zero diagonal

deg = adj.sum(axis=1)
d_inv = np.diag(1.0 / deg)

lap_unnormalized = sp_laplacian(adj, normed=False)
lap_normalized = sp_laplacian(adj, normed=True)
lap_random_walk = np.eye(n) - d_inv @ adj

# Reference invariant subspace: the k smallest eigenvectors of L_sym.
evals, evecs = np.linalg.eigh(lap_normalized)  # ascending eigenvalues
eig_subspace = [evecs[:, c].tolist() for c in range(k)]  # k columns, each length n
gap = float(evals[k] - evals[k - 1])  # separation that makes the k-subspace well defined

fixture = {
    "provenance": {
        "generator": "gen_lapl.py",
        "library": "scipy + numpy",
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "seed": SEED,
        "eigenvalues": evals.tolist(),
        "subspace_gap": gap,
        "note": "matrices are scipy/numpy oracles; eigenvectors compared as a subspace.",
    },
    "k": k,
    "adj": adj.tolist(),
    "expected": {
        "lap_unnormalized": np.asarray(lap_unnormalized).tolist(),
        "lap_normalized": np.asarray(lap_normalized).tolist(),
        "lap_random_walk": lap_random_walk.tolist(),
        "eig_subspace": eig_subspace,
    },
}

out = Path(__file__).parent / "lapl_spectral.json"
out.write_text(json.dumps(fixture, indent=2) + "\n")
print("eigenvalues", [f"{v:.6f}" for v in evals])
print(f"subspace gap at k={k}: {gap:.6f}")
print(f"wrote {out}")
