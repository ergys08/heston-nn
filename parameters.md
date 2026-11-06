# Experiment parameter map

For the permanent module/folder responsibilities and planned historical loop,
see [ARCHITECTURE.md](ARCHITECTURE.md).


**Change experimental settings only in [main.py](main.py) or
[main_random.py](main_random.py), depending on the experiment.** Each run passes
those values explicitly to `build_distributions`. No experiment result files
are written; the returned `DistributionBatch` stays in memory.

## Experiment settings

| Parameter in `main.py` | Meaning | Where it is used |
|---|---|---|
| `N_DATES` | Number of historical observations (`None` = all) | `main.py: main()` slices the calibration list |
| `N_QUBITS` | Output resolution: `2**N_QUBITS` bins | `main.py → build_distributions() → discretize_pdf()` |
| `CUTOFF` | Symmetric log-return interval `[-cutoff, cutoff]` | `build_distributions() → discretize_pdf()`: bin edges and retained probability |
| `N_FFT` | Fourier inversion grid size | `build_distributions() → heston_pdf()`: frequencies and PDF resolution |
| `FFT_HALFWIDTH` | Fourier inversion domain `[-halfwidth, halfwidth)` | `heston_pdf()`: spatial grid and FFT shift |
| `MATURITY` | Terminal horizon in years | `heston_pdf() → characteristic_function()`: Heston time evolution |
| `DRIFT` | Drift of **asset price**, not log-return drift | `characteristic_function()`: factor `exp(i*u*drift*maturity)` |
| `VERIFY_FFT` | Check discretised masses at doubled Fourier resolution | `build_distributions()`: optional second PDF calculation |
| `FFT_L1_TOLERANCE` | Maximum sum of absolute bin-mass differences in FFT refinement check | `build_distributions()`: numerical acceptance threshold |

## Five Heston model parameters

The historical data are stored unchanged in
[`heston/heston_params_2013.csv`](heston/heston_params_2013.csv).

| Model parameter | CSV column | Location in code | Role |
|---|---|---|---|
| Initial variance `v0` | `v0` | `read_calibrations() → HestonParameters → characteristic_function()` | Initial instantaneous variance |
| Mean reversion `kappa` | `kappa` | Same route | Speed of variance mean reversion |
| Long-run variance `theta` | `theta` | Same route | Long-term variance level |
| Volatility of variance `xi` | **`sigma`** | Same route | Vol-of-vol; the CSV name is not asset volatility |
| Brownian correlation `rho` | `rho` | Same route | Correlation between price and variance innovations |

Every date supplies its own five model parameters. The nine numerical experiment
settings in `main.py` are shared across dates.

## Code flow

```text
main.py (experiment settings)
  |
  +--> read_calibrations()                 [heston/build_distributions.py]
  |       |
  |       +--> heston_params_2013.csv
  |       +--> HestonParameters             [heston/distributions.py]
  |
  +--> build_distributions()               [heston/build_distributions.py]
          |
          +--> heston_pdf()                [heston/distributions.py]
          |       +--> characteristic_function()
          |
          +--> discretize_pdf()            [heston/distributions.py]
          |
          +--> optional FFT refinement check
          |
          +--> DistributionBatch in memory (no files created)
```

## Numerical safeguards

`heston/distributions.py` includes fixed safeguards for valid Heston
parameters, PDF nonnegativity, PDF normalisation, FFT boundary mass, valid
bin edges, and normalised conditional probabilities. These are **numerical
checks**, not additional experimental parameters. The two current fixed PDF
thresholds are negative FFT mass `1e-8` and outer-grid mass `1e-5`.
If these thresholds become part of a sensitivity study, promote them into
`main.py` too.

The stored `bin_masses` are *unconditional* probabilities, whereas
`probabilities` are renormalised to the truncation interval. Use
`retained_masses` to quantify probability lost beyond the cutoff.

## Where to add future parameters

When adding a transport experiment, define the new settings in the labelled
block in `main.py`, pass them as explicit function arguments, and add an
entry to this map. Avoid defining separate experimental defaults throughout the
implementation.

## Transport architectures and chronological loop

**Both fixed architectures are run on every experiment**, with 16 and 32
ordered Pauli rotations respectively. There is no budget selector.
The lists are in [`circuits/generators.py`](circuits/generators.py), and
`logic/experiment.py: run_experiments()` runs them independently on the
same historical Heston states and with the same optimiser settings.

| Setting in `main.py` | Meaning | Used by |
|---|---|---|
| `MINIMUM_FIDELITY` | Minimum squared amplitude overlap for an accepted transport | `logic/experiment.py` and `fit_angles()` |
| `MAX_ITERATIONS` | Maximum L-BFGS-B iterations **per starting point** | `logic/optimisation.py` |
| `EXTRA_RESTARTS` | Additional seeded, perturbed optimisation starts | `fit_angles()` |
| `RANDOM_SEED` | Reproducible numerical seed for those extra starts; offset by date index | `fit_angles()` |
| `PROGRESS_EVERY` | Print per-architecture progress after every N dates; 0 prints only summary | `logic/experiment.py` |
| `N_QUBITS` | Fixed at **8** for these archived word sets | Heston builder and `generator_sets()` |

No `REFERENCE_INDEX` is offered: the initial historical date always becomes
the first reference. Every later date attempts a transport from the
**currently active reference**, not from the previous transported result.
If gate-simulated fidelity is below `MINIMUM_FIDELITY`, that date is
prepared from scratch as a new reference; the same frozen ordered Pauli words
remain in use for its architecture. The two architectures can change
references on different dates.

### Implemented code flow

```text
main.py
  ├── heston/build_distributions.py
  │     └── Heston probabilities, amplitudes, retained mass, dates
  └── logic/experiment.py: run_experiments()
        ├── generator_sets() -> BOTH 16 and 32
        ├── circuits/preparation.py -> exact reference loader
        ├── logic/optimisation.py -> current-target oracle L-BFGS-B
        ├── circuits/transport.py -> ordered exp(-i angle * Pauli) compilation
        └── circuits/gates.py -> independently simulate elementary gates
```

`main()` returns `data, runs` with `runs[16]` and `runs[32]`.
Each `ExperimentRun` contains `daily` (one dict per date),
`references` (date/index/active region/preparation cost) and
`summary` (transport count, reference count, fidelity and cost totals).
For each daily record:

- `decision`: `new_reference` or `transport`
- `reference_date` and `reference_index`: actual active reference **after**
  the decision; `attempted_reference_date/index`: pre-decision reference
- `attempted_transport_fidelity`, `output_fidelity`: numerical target
  attempt and final realised circuit state overlap
- `retained_mass`, `lost_mass`, `retained_weighted_fidelity`: truncation
  error kept separate from conditional fidelity
- `angles`: accepted rotations on transport days, `None` on reference
  preparation days; `attempted_angles`: candidate even if rejected
- `optimiser_starts/iterations/evaluations`: classical fitting work
- `attempted_transport_cost`, `preparation_cost`,
  `executed_logical_cost`: attempted gate budget is recorded even on reset
  days; only *executed* preparation or successful transport contributes to
  the reported logical totals.

### Scientific limitations

The original first-reference (2013-01-02) Pauli sets came from historical
tangent selection at cutoff +/-0.7. **No tangent calculations now occur.**
In the old replay, different words were selected after each reset; the
current experiment deliberately retains each initial frozen set across all
resets. Current cutoff +/-2.5 also differs. Old historical reference counts
are therefore not automatically reproducible.

The optimisation has access to each current target state, so it is an
**oracle expressivity benchmark**, not an efficient neural-network parameter
predictor. Failure to reach the threshold is an optimiser-dependent reset,
**not proof that the architecture cannot represent the target**. Neither
a neural network nor a hardware-noise model has been implemented.

Gate counts include from-scratch preparation when a new reference is made
and a complete compiled single-pass transport on accepted days. They assume
arbitrary-pair CNOT connectivity and include no routing, measurement,
training or classical optimisation costs. Rejected proposed transport
circuits are logged but *not* charged as executed gates.

No output files are written. Numerical arrays, dates, attempts and decisions
are retained in memory; the script prints a compact paired summary.
Install dependencies using `python3 -m pip install -r requirements.txt`.


## Random-path comparison: `main_random.py`

Run `python3 -B main_random.py` to compare the 2013 historical order to
seeded permutations of **exactly the same** encoded target states. Every
permutation keeps observation 0 fixed and visits every other observation
exactly once. **Both** frozen 16- and 32-generator circuits are evaluated on
the true chronology and on **every** random path, with independent active
references and no tangent-based generator selection.

The separate entry point duplicates only the numerical settings at the top,
not the fitting or chronological decision algorithms. Keep its shared
Heston/optimiser settings equal to `main.py` when comparing experiments.
`main_random.py` currently uses `MAX_ITERATIONS=40` to match the previous
250-date random-order benchmark (the normal `main.py` default is 250).
Do **not** compare runs across different iteration budgets as if they were
methodologically identical.

| Random-path setting | Purpose |
|---|---|
| `N_PERMUTATIONS` | Number of independent random orderings of observations 1...T-1 (default 20) |
| `PATH_SEED` | Reproducible NumPy permutation seed (default 20261102) |
| `PROGRESS_EVERY` | Print status every N completed permutations; zero disables progress |

`logic/random_paths.py: compare_random_paths()` first executes the real
chronology with `run_experiments()`, then shuffles all date-dependent fields
consistently using the same chronological algorithm and optimiser. It returns
a `RandomPathComparison` with:

- `chronological`: `{16: ExperimentRun, 32: ExperimentRun}`
- `random_trials`: each trial contains its original observation-index
  permutation, mean successive Fubini–Study distance, and both complete
  `ExperimentRun` objects (daily decisions, references and optimiser metrics)
- `summary`: historical and random reference counts, logical CNOTs,
  mean successive state-space distances and one-sided permutation p-values,
  computed as `(1 + count(random_refs <= historical_refs))/(1 + n_permutations)`.

The control tests whether **historical ordering** reduces online reference
preparations. Because the state collection is unchanged, it says nothing by
itself about whether the underlying Heston family has a smaller intrinsic
geometric covering number than other families. It also cannot disentangle
state geometry from the optimiser's previous-angle warm start. The 20-path
default is computationally demanding; use `N_PERMUTATIONS=2` and
`N_DATES=12` for a short smoke test.

Like the chronological main, this runner writes **no output files**. All
results are returned in memory.
