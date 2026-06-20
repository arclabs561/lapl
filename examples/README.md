# lapl examples

Examples for graph Laplacians and spectral diagnostics.

## Running

```sh
cargo run --example spectral_rmt
```

Use `cargo test --examples` to compile every example.

## Task map

| Goal | Example | What to inspect |
|---|---|---|
| Compare random and clustered graph spectra | `spectral_rmt` | The low end of the normalized Laplacian spectrum. The random graph has one small eigenvalue, while the three-cluster graph has three small eigenvalues followed by a large eigengap. |

## Reading path

Start with the printed eigenvalue summaries before reading the histogram. For
community structure, the small eigenvalues and their gap carry the main signal.
The RMT spacing and density summaries are useful context for how the rest of
the spectrum behaves.
