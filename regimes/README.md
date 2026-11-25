# Synthetic Heston entanglement regimes

This experiment is independent of historical calibrations. It uses the existing
Heston Fourier inversion / bin integration and the existing q0-first right-to-left
MPS TT-SVD. No quantum circuit optimisation, hardware or market data is required.

## Run locally

From the repository root, in your usual Python environment:

    python -m pip install numpy scipy matplotlib
    python -m unittest regimes.test_regimes
    python -m regimes.run_experiment --points 3 --vary a b --name smoke
    python -m regimes.run_experiment --name baseline

The baseline runs all ten pairwise slices, holding the other three coordinates
at the same reference (a,b,c,d,rho) = (1,0.1,1,0.04,-0.7). It includes the
reference and the a=1, b=1, rho=0 landmarks exactly and deduplicates shared
models. The default 13-point grids plus landmarks are deliberately exploratory,
not a distribution of calibrated market parameters.

| Coordinate | Definition | Default range |
| --- | --- | --- |
| a | kappa*T | 0.01–100, log |
| b | xi^2/(2*kappa*theta), inverse Feller ratio | 0.001–10, log |
| c | v0/theta | 0.1–10, log |
| d | theta*T | 0.0025–0.25, log |
| rho | Brownian correlation | -0.99–0.99, linear |

The Feller boundary is b=1. Both sides are intentionally explored. v0 is fixed
(deterministic) within each model, not sampled from the stationary variance law.
All five dimensionless quantities can be varied. Pairwise slices do not cover
all higher-order interactions; use a restricted full Cartesian grid if needed:

    python -m regimes.run_experiment --mode full --vary a b rho --points 9 --name abrho
    python -m regimes.run_experiment --points 7 --qubits 10 --vary a b --name n10
    python -m regimes.run_experiment --points 7 --qubits 12 --natural-only --vary a b --name n12
    python -m regimes.run_experiment --points 7 --vary a b --cutoff 8 --name wider_encoding
    python -m regimes.run_experiment --help

Full 5D grids can be enormous; a model-count guard prevents accidental large
runs. At 8 qubits the all-cuts option computes 127 unique bipartition spectra;
it is enabled by default. For >10 qubits, natural-only is required.

## Encoding and numerical checks

At fixed T (default one year), convert to kappa=a/T, theta=d/T,
xi=sqrt(2*kappa*theta*b), v0=c*theta. Standardise each terminal log return
using its exact mean and variance from the closed linear moment equations.
Set constant log-price drift contribution to zero because it cancels under
standardisation. Encode conditional bin masses on the common [-6,6] grid
(default), not pointwise PDF values. Each probability amplitude is sqrt(p).
Qubit q0 is the least significant bit. This is a finite, conditional encoding:
retained probability mass is always reported; cutoff changes can affect spectra.

Every state is checked at increased FFT resolution and at doubled physical
domain with equal fine spacing, comparing encoded distributions using
squared-overlap infidelity and retained mass. Untruncated mean and variance
are checked independently against analytic moments. Numerical failures are
recorded, not replaced with Gaussian states or silently skipped. Increase
--max-fft for extreme models. Convergence comparisons are not certified
error bounds; re-run interesting boundaries with tighter tolerances, larger
cutoff and different qubit counts. Extreme synthetic parameters may fail.
Default numerical infidelity comparison tolerance is 1e-8, two orders below
the smallest default target infidelity. The original repo's FFT boundary and
negative-density checks still apply.

## Results

Each run creates a new directory regimes/results/<name>/ (or UTC timestamp).
Existing runs are never overwritten.

* config.json: exact grids, settings, reference and software versions.
* models.csv: all cases including numerical failures, raw Heston parameters,
  retained mass, FFT convergence diagnostics, exact mean/std, and fixed-v0
  integrated variance relative variance Var(A)/E(A)^2.
* states.npz: successful case_ids, probabilities, amplitudes, common bin edges
  (unless --no-states).
* schmidt_spectra.csv: each bipartition's squared Schmidt coefficients
  (JSON array, descending), entropy in bits, physical qubit list, and whether
  it is a natural contiguous cut. Schmidt singular values are sqrt(weights).
  Complementary cuts are stored only once.
* ranks.csv: one row per model and target infidelity. natural_profile lists the
  minimum rank at each natural cut; natural_cut_bound is their maximum.
  natural_tt_cap is the first integer cap achieving the target global fidelity
  using the repository's TT-SVD. best_cut_bound is the exact minimax cut lower
  bound over all qubit permutations, computed using subset DP. best_cut_order
  is one minimax ordering (lexicographic tie break). reordered_tt_cap measures
  global fidelity on that chosen order. best_tested_tt_cap is the better of
  natural and chosen-order TT-SVD; it is NOT the globally optimal MPS cap.
  cut_order_gap is natural_cut_bound minus best_cut_bound.
* mps_fidelities.csv: full fidelity-versus-bond-cap curves for the natural
  order and each selected reordered chain.
* figures/: bond-dimension heatmaps for each pairwise slice and each target
  infidelity. Grey cells indicate failures or missing observations. Red crosses
  identify the reference. On full grids the plots still show reference slices.
* summary.json: attempted/successful/failed counts for a completed run.

CSV outputs are flushed per model. On normal completion (or Python-level
interrupt), successfully collected state arrays are saved. No resume support:
use a new run name. Large all-cuts runs are intentionally computationally
expensive; start small.

## Interpreting structural changes

For a target infidelity eps, the smallest rank at a cut is the first r for
which the sum of squared Schmidt coefficients beyond r is <= eps. The maximum
over cuts is a necessary lower bound on any MPS bond cap achieving global
fidelity >= 1-eps. Actual TT-SVD fidelity is computed separately; the SVD
construction is not globally optimal.

A higher natural cut bound with an unchanged best-order cut bound suggests
an ordering effect. An increase even in the best-order bound is evidence
of greater linear-MPS rank requirements at that tolerance. A large bond
dimension does not prove that a tree network is cheaper: this experiment
does not optimise TTNs or establish a phase transition. Discrete rank
changes depend on approximation tolerance, binary encoding, and resolution.
Check spectra and global fidelity curves near any boundary before drawing
conclusions. Full five-dimensional claims require additional joint sampling.

Suggested next step: inspect heatmaps for a,b and b,rho, locate rank changes,
then use higher --points and repeat with different reference, cutoff and qubits.
