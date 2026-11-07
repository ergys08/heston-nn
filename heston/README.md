# Historical Heston calibration data (2013)

## Raw data

- `heston_params_2013.csv`: **unmodified** copy of the published daily calibration CSV.
- Upstream repository: [oskarimikkila/Empirical-Deep-Hedging](https://github.com/oskarimikkila/Empirical-Deep-Hedging)
- Upstream file: [data/heston_params.csv](https://github.com/oskarimikkila/Empirical-Deep-Hedging/blob/master/data/heston_params.csv)
- Upstream Git blob SHA: `1240fa9f324b5a7d6319657e673591007d43c59a`
- Number of daily observations: **250**
- Date range: **2013-01-02** through **2013-12-31**
- Original column order: `date,theta,kappa,sigma,rho,v0`

## Parameter conventions

- `v0`: initial instantaneous variance
- `theta`: long-run variance
- `kappa`: mean-reversion rate
- `sigma`: volatility of variance (**vol-of-vol**, denoted `xi` in the transport code)
- `rho`: Brownian-motion correlation

Do not confuse `sigma` with asset volatility; it is Heston's vol-of-vol parameter.

## Reproducibility policy

The CSV is preserved as raw source data, including its original headers and numerical precision. Any renaming, filtering, cleaning, or parameter transformation should happen in a **separate, tested loader** rather than by modifying this file.

These are previously published model calibration estimates. This repository does not claim to have calibrated the original option data or independently validated the quality of each calibration. Estimates near calibration constraints require explicit diagnostics before economic interpretation.

## Additional 2020 calibration panel

- `heston_params_2020.csv`: unmodified copy of [nlal46/Heston-Numerical-Calibration](https://github.com/nlal46/Heston-Numerical-Calibration/blob/main/calibration_results_2020.csv).
- **71** trading dates from **2020-01-02** to **2020-04-14**, including the onset of the COVID-19 market disruption (not the full calendar year).
- Original columns: `date,v0,kappa,theta,eta,rho,sqrt_v0,sqrt_theta,feller,loss,n_options,degenerate`.
- Original 2020 header `eta` means Heston vol-of-vol and is mapped to `xi` in code (the 2013 source calls it `sigma`).
- **3** dates are marked `degenerate=True`: **2020-03-18**, **2020-03-20**, and **2020-03-26**. The raw CSV preserves all rows. The loader excludes them only on explicit request.
- Unlike the 2013 panel, these calibrations were produced with a different fitting procedure (vega-weighted price RMSE, differential evolution followed by L-BFGS-B). Differences between years may reflect calibration methodology as well as market regimes. The upstream option-chain CSV is not bundled with the source project.

## Constructing quantum states and covering analysis

The same PDF inversion, eight-qubit discretisation, Fubini--Study fidelity, and step-length-matched random-walk utilities apply to both years. Default commands preserve the original 2013 workflow:

```bash
python -m heston.compare_covering --year 2013 --n-walks 100
python -m heston.compare_covering --year 2020 --n-walks 100
python -m heston.compare_covering --year 2020 --exclude-degenerate --n-walks 100
```

The 2020 covering command defaults to an FFT grid halfwidth of **25** and **131072** grid points (verified against 262144), rather than the 2013 defaults of halfwidth **3.5** and **16384** points. This is necessary because extreme crisis-era densities have much more probability near the edges of the narrow Fourier inversion domain. The quantum encoding window remains **[-2.5, 2.5]** for both years. Override using `--fft-halfwidth` and `--n-fft`; do not disable verification merely to silence the boundary check. The 2020 defaults are more computationally expensive. The program saves `retained_masses.csv` to document how much original probability lies inside the encoded interval on each date. These settings have been checked on representative 2020 calibrations, not yet validated across every observation.

Default output locations are `covering_results/` (2013) and `covering_results_2020/` (2020). Use `--output-dir` to avoid overwriting results when comparing filtered and unfiltered 2020 samples. The summaries include counts, counts divided by the number of states, and the Heston/random-walk covering ratio. The default greedy method returns a **valid sample-centred cover**, not a certified minimum.

For apples-to-apples geometric comparisons, also compare both years using equally sized chronological windows (e.g. 68 or 71 states) and the same fidelity thresholds. A fraction `count / N` removes the trivial difference in the number of sampled states but does *not* completely remove dependence on sampling density or differing step-size distributions. Inspect retained PDF mass and inversion diagnostics especially for extreme 2020 parameters.
