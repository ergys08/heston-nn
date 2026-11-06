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
