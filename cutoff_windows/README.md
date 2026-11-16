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

For a useful balanced benchmark, try `F_target=0.99835`: 355 one-layer versus 150 two-layer cases. This is **conditional preparation fidelity on the standardised truncated state**; if you need end-to-end fidelity against the full distribution, account separately for retained mass.

The sweep output is descriptive, not a proof that a single scalar generalises to different Heston regimes or other models. The 2019 versus 2020 split is one chronological holdout and the daily calibrations are dependent.

## Workflow-level cost–fidelity comparison

The script `cutoff_windows/compare_workflows.py` directly compares **three preparation policies** using the already saved one- and two-layer conditional MPS fidelities:

1. Always use one MPS layer.
2. Always use two MPS layers.
3. Select one or two layers using the three-sigma tail-weight cutoff learned from **2019 only**, with **2020 as an unseen holdout**.

Run from the repository root after pulling the latest code:

```bash
git pull origin main
python -m cutoff_windows.compare_workflows
```

Outputs, overwritten on each run, are `workflow_comparison_daily.csv` (every date and policy), `workflow_comparison_summary.csv` (aggregated measures for 2019, 2020 and full dataset), and `workflow_comparison_summary.json` (configuration, held-out comparisons, savings and caveats). No Heston refitting, state recalculation or MPS optimisation is needed.

The default target is **0.99835 conditional-state fidelity** on the eight-qubit *standardised* window. Change it with `--target 0.9984`, or try a kurtosis predictor with `--feature excess_kurtosis`. Both predictors use the same conservative 2019-only training rule: use one layer only if the predictor exceeds the largest predictor measured among 2019 states that fail with one layer.

One MPS layer has `N-1=7` native two-qubit blocks; two layers have 14. The script reports **nominal** CNOT counts, assuming two CNOTs per real two-qubit block (14 versus 28 CNOTs). This is a cost proxy derived from the existing MPS implementation, **not** a Qiskit transpilation measurement. Feature computation, classical optimisation and hardware runtime are not counted. Importantly, *always one layer has the lowest gate cost*, and *always two layers has the highest fidelity*: the selector's success criterion is meeting the **required fidelity at lower cost than always two layers**, rather than dominating both baselines on every metric.

On the **2020 holdout** for `F>=0.99835`, the three-sigma selector uses one layer on 229/253 days, reaches the target on all 253, and uses 3,878 nominal CNOTs in total (15.328 per state). Always one layer uses 3,542 CNOTs but fails the conditional fidelity target on nine days. Always two layers uses 7,084 CNOTs and meets it on every day. Thus the selector saves approximately **45.3%** of the two-layer CNOT budget at the same target pass rate; the counts are conditional on this Heston panel and the standardised encoding convention.
