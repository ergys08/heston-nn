# Heston tail-shape versus MPS-complexity experiment

The experiment analyses the same **505 actual SPX-return-fitted Heston calibrations** (2019-01-02 to 2020-12-31). Return-based estimates are under the physical measure, **not** the original option-implied risk-neutral measure.

## Run locally

From the root of `heston-nn`:

```bash
git pull origin main
python -m pip install numpy scipy matplotlib
python -m unittest cutoff_windows.test_run_experiment -v
python -m cutoff_windows.run_experiment \
  --input heston/SPX_generated_heston_params_2019_2020.csv \
  --output-dir cutoff_windows
```

**The script overwrites the existing** `cutoff_windows/daily_results.csv`, `correlations.csv`, `summary.json`, `timeline.png`, and `correlation.png`. It also refreshes `failures.csv`. It never modifies the original Heston calibration CSV. A short smoke test with separate outputs:

```bash
python -m cutoff_windows.run_experiment --max-states 5 --output-dir cutoff_windows/smoke
```

## Main question: does genuine tail shape predict MPS complexity?

The original experiment defined tail weight as probability outside a **fixed raw-return interval**. That variable was almost perfectly correlated with return variance, and so was not a clean measure of heavy-tailedness.

The revised experiment uses **two full-distribution, location/scale-invariant predictors**:

- **Excess kurtosis**: `E[((X-mu)/sd)^4] - 3`. Gaussian baseline = 0.
- **Standardised tail mass**: `P(|X-mu| > 3 sd)`, plus thresholds at 2 and 4 standard deviations. A Gaussian's three-sigma tail mass is approximately 0.00270. For a skewed Heston density this combines left and right tails; skewness is also recorded separately.

These statistics are calculated from the *untruncated* Fourier-inverted Heston density, not from the conditionally renormalised state.

### Three preparation domains (all use the same qubit count)

1. **`standardized` (primary analysis):** On the first date, choose the smallest interval symmetric around the mean in **standard deviation units** containing 99.9% of that date's probability. Hold that dimensionless radius `z0` constant for every date and encode each density on `[mu_t-z0*sd_t,mu_t+z0*sd_t]`. Thus each state has exactly the same z-domain and z-bin width, eliminating changes in location and scale.
2. **`fixed` (legacy control):** Keep the first date's symmetric *raw log-return* interval `[-R0,R0]` unchanged over time.
3. **`adaptive` (legacy control):** On each date, choose its own smallest symmetric *raw log-return* interval containing 99.9% probability.

`standardized_tail_mass_3sigma` is the probability beyond **three standard deviations**, regardless of the narrower/wider cutoff used for quantum encoding. `standardized_tail_mass` and kurtosis are *predictors*, whereas MPS bond dimensions and circuit fidelities are outcomes.

## Results written

- `daily_results.csv`: date, model parameters, full-distribution mean/SD/variance/skewness/excess kurtosis, standardised tail masses (2/3/4 sigma), all three cutoff windows' retained masses and widths, exact Schmidt ranks, discarded `chi=2` weights, MPS-bond entropies, minimum TT-SVD bond dimension at the configured fidelities, and iterative MPS fidelities for 1–3 layers.
- `correlations.csv`: Pearson and Spearman correlations, with `sample` = `all`, `2019`, or `2020`, and `analysis` = `levels` or `first_difference`. First differences avoid crossing the year boundary. Undefined correlations are blank.
- `summary.json`: configuration and highlighted shape/complexity correlations.
- `timeline.png`: chronological kurtosis, standardised tail probability and one-layer fidelities.
- `correlation.png`: full-distribution excess kurtosis against standardised-state Schmidt truncation, effective rank and one-layer fidelity.
- `failures.csv`: failed calibration dates, if any.

`*_chi_for_995` refers to the TT-SVD approximation at fidelity 0.995; `*_layers_for_995` is the number of **tested iterative circuit layers** required for the same fidelity, or blank if not reached. Numerical maximum Schmidt rank is often uninformative because it saturates.

## Limitations

Excess kurtosis is not the only definition of heavy tails: it is sensitive to the fourth moment and may also reflect features outside the far tails. Standardised tail probability at 3 SD is a complementary, direct measure of tail weight at a fixed z threshold. Skewness and the interplay of left/right tails can also matter.

All the state targets are conditional on the chosen encoding window. When the standardised window retains different fractions across dates, preparation fidelity refers to the **conditional quantum state**, not the full real-line density. A distribution's standardisation does not guarantee identical MPS difficulty. Successive market calibrations are highly autocorrelated, so naive significance tests or a pooled correlation should not be interpreted as independent evidence or causation. Comparing 2019 and 2020 separately, and using first differences, is more informative.

To change fidelity: `--fidelities 0.99 0.995 0.999`; qubits: `--qubits 10`; iterative layers: `--layers 4`; first-date retained probability: `--retained-fraction 0.9999`.

## Selecting one versus two MPS layers: fidelity sweep

The original fidelity targets (`0.995` and `0.999`) rarely distinguish one from two MPS layers. The new **one-versus-two classifier** checks thresholds spanning `0.99820` through `0.99900`, without requiring new circuit optimization.

**Fast mode — recompute only classification from already saved daily fidelities:**

```bash
git pull origin main
python -m cutoff_windows.run_experiment --analyse-existing
```

This overwrites `cutoff_windows/summary.json` and writes `cutoff_windows/threshold_sweep.csv`. It leaves the original daily results and plots intact. A full rerun using `python -m cutoff_windows.run_experiment` also produces this new sweep. The scan does **not** change the existing `--fidelities` MPS decomposition targets; it reuses the already recorded one- and two-layer fidelities for finer thresholds.

Each threshold reports: (i) how many of the 505 standardised states satisfy `F_1 >= F_target`, (ii) whether two layers reach the target, (iii) ROC AUCs for kurtosis and three-sigma tail mass over all dates and separately within 2019/2020, and (iv) a conservative rule fitted **only on 2019**, evaluated on unseen **2020** dates. The cutoff rule chooses one layer only when the tail-shape predictor exceeds the largest 2019 predictor value for a state requiring two layers. It therefore makes no *training* false-positive one-layer recommendations by construction, but 2020 holdout violations remain possible and are explicitly counted.

The default representative fidelity target is now **0.99842**, at which 16 of 253 held-out 2020 states require two layers (rather than nine at 0.99835). This leaves 16 positive one-layer examples in 2019. Thresholds approaching 0.99845 yield only two positive 2019 examples, and thresholds >= roughly 0.99848 lack any; these are unsuitable for calibrating this one-year selector. This is **conditional preparation fidelity on the standardised truncated state**; if you need end-to-end fidelity against the full distribution, account separately for retained mass.

The sweep output is descriptive, not a proof that a single scalar generalises to different Heston regimes or other models. The 2019 versus 2020 split is one chronological holdout and the daily calibrations are dependent.

## Workflow-level cost–fidelity comparison

The script `cutoff_windows/compare_workflows.py` compares **four policies** using the *same previously saved* MPS fidelities:

1. Always use one MPS layer.
2. Always use two MPS layers.
3. Three-sigma-tail-based selector, trained on 2019.
4. Excess-kurtosis-based selector, independently trained on 2019.

### Rerun both experiments and overwrite previous outputs

From the repository root, run:

```bash
git pull origin main
python -m cutoff_windows.run_experiment --analyse-existing
python -m cutoff_windows.compare_workflows
```

The first command re-evaluates both selectors' classification/ROC-AUC diagnostics on the extended fidelity sweep and **overwrites** `threshold_sweep.csv` and the existing selector-sweep section of `summary.json`. Its representative fidelity is now **0.99842**. The second evaluates all four workflows on the same stricter default target and **overwrites** the existing `workflow_comparison_daily.csv`, `workflow_comparison_summary.csv` and `workflow_comparison_summary.json` (no new output names). The source `daily_results.csv`, `correlations.csv` and plots remain untouched: no state-preparation simulation is required. Both selectors are fitted on 2019 only and tested on held-out 2020.

For a different threshold, use a matching pair of commands:

```bash
python -m cutoff_windows.run_experiment --analyse-existing --sweep-thresholds 0.99845
python -m cutoff_windows.compare_workflows --target 0.99845
```

At **0.99842**, 16 of the 253 2020 states require two MPS layers; 2019 still contains 16 one-layer-sufficient cases, allowing a nondegenerate training rule. At 0.99845, 19 2020 states need two layers but only **two** 2019 states are one-layer sufficient; the resulting selection rule has very limited training support. Beyond roughly 0.99848, 2019 contains no one-layer successes, so the comparison script refuses to train a meaningful 2019-only selector at those targets. The conservative rule selects one layer only when the feature value is above the largest value observed among *2019 one-layer failures*.

Nominal CNOT counts use **two CNOTs per real two-qubit MPS block**: 14 for one 8-qubit MPS layer and 28 for two. These are nominal structural estimates, not actual Qiskit-transpiled costs. The experiment measures **conditional fidelity on the standardised truncated state** and excludes the classical cost of the predictor and neural-network inference. Fidelity-threshold exploration uses information from the 2020 data, so 2020 is a chronological holdout for the *fitted selector cutoffs*, but not a fully untouched dataset for threshold selection. Follow-up testing on another historical period or parameter panel is necessary for an unbiased performance estimate.
