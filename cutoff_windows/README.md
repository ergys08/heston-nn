# Fixed-cutoff versus adaptive-cutoff Heston MPS study

This experiment analyses the **505 real SPX-return-calibrated Heston parameter sets**, 2019-01-02 through 2020-12-31, in `heston/SPX_generated_heston_params_2019_2020.csv`. It uses the existing `heston` Fourier inversion and `MPS` preparation modules. These are **return-fitted** Heston calibrations under the physical measure, not the original risk-neutral/option-fitted 2013 parameters.

## Run (from repository root)

```bash
git pull origin main
python -m pip install numpy scipy matplotlib
python -m unittest cutoff_windows.test_run_experiment -v
python -m cutoff_windows.run_experiment \
  --input heston/SPX_generated_heston_params_2019_2020.csv \
  --output-dir cutoff_windows
```

The default full run analyses 505 trading dates. For a quick smoke test that does not overwrite full-run outputs:

```bash
python -m cutoff_windows.run_experiment --max-states 10 --output-dir cutoff_windows/smoke
```

The full run writes **all outputs into `cutoff_windows/`**:

* `daily_results.csv` — dated parameters, fixed and adaptive cutoffs, probability lost, variance, excess kurtosis, seven-bond Schmidt ranks, discarded Schmidt weights, effective MPS bond dimension at fidelities 0.99, 0.995 and 0.999, and actual iterative 1–3-layer MPS fidelities.
* `correlations.csv` — Pearson and Spearman coefficients comparing tail mass, adaptive cutoff radius and shape parameters against MPS complexity measures.
* `summary.json` — exact run settings and selected correlations.
* `timeline.png` and `correlation.png` — visualisations.
* `failures.csv` — failed PDF/spectral checks, if any.

## Cutoff selection

The **first date** determines a symmetric interval `[-R0,+R0]` containing 99.9% of its full terminal log-return probability. The same `R0` is used for every date. The fixed-window tail weight is `1 - P(|X|<=R0)`, *before renormalisation*.

For comparison, each date gets an **independently selected smallest symmetric** `[-Rt,+Rt]` retaining 99.9% of its own probability. Both windows are converted to **8-qubit quantum states (256 probability-bin amplitudes)**, using precisely your existing Heston PDF discretisation code. The adaptive window's tail mass is ~0.001 by construction, so use the fixed-window mass loss or adaptive radius as predictors.

Default physical settings: maturity = 1 year, drift = 0.05, FFT halfwidth = 25, FFT points = 131072, 8 qubits. The larger FFT domain helps prevent falsely reporting probability loss because the Fourier inversion domain is too narrow.

## MPS complexity measures

`*_max_exact_bond_dimension` is the maximum **numerical Schmidt rank** among seven qubit bipartitions; numerical ranks often saturate at full rank.

More useful quantities are `*_max_discarded_chi2` (largest across-bond probability weight beyond the first two Schmidt coefficients), `*_chi_for_995` (smallest allowed TT-SVD bond dimension giving fidelity >=0.995), and `*_layers_for_995` (smallest **tested** iterative MPS layer count reaching fidelity >=0.995; blank if three layers do not suffice). The number of layers is a *measured result*, not equal by definition to an MPS bond dimension.

## Interpretation

Tail loss and MPS difficulty need **not** be positively correlated. Cutting off heavy tails changes and renormalises the prepared state, and may remove complex tail structure. The fixed and adaptive windows also have **different bin widths**. Mass loss can arise from higher variance even without larger standardised kurtosis. Treat correlation as observational, especially because successive calibrated parameters are temporally dependent.

Override the 99.9% mass requirement with `--retained-fraction 0.9999`, change `--layers 4`, or compare `--qubits 10`. The cutoff is always selected from the first date of the input (or subset), not manually specified.
