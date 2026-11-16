# Calibrating a Heston model to genuine S&P 500 historical returns

This module produces **real-data-based Heston calibrations**, not simulated parameters. It uses recorded daily closes of the **S&P 500 price index** (SPX, Yahoo symbol `^GSPC`), matching the underlying index discussed in the 2013 *Empirical Deep Hedging* source.

**Important difference:** `heston_params_2013.csv` was derived from option-price calibrations. This new estimator fits a Heston process to observed **underlying index returns**, so its parameters are estimated under the physical probability measure, not the option-pricing/risk-neutral measure. Even for identical dates, the estimates are not interchangeable. Real returns alone also provide relatively weak information about all Heston parameters. Use these calibrations for exploratory robustness testing of quantum state-preparation methods, **not** as a verified extension of the original risk-neutral calibration series.

## Setup and tests

From the root of `heston-nn`:

```bash
python -m pip install numpy scipy yfinance
python -m unittest heston.test_calibrate_spx_returns -v
```

The estimator attempts to download real SPX closes from Stooq first, then from Yahoo Finance using `yfinance`. If both fail, it stops with an error; it **never** invents market observations.

## Generate daily 2014 calibrations

```bash
python -m heston.calibrate_spx_returns \
  --start 2014-01-02 --end 2014-12-31 \
  --history-start 2010-01-01 \
  --lookback 504 --refit-every 21 \
  --output heston/generated/spx_returns_heston_2014.csv
```

This generates an estimate on **each observed SPX trading date**. The four relatively persistent structural coefficients (`kappa,theta,sigma,rho`) are fitted every 21 trading days using the preceding 504 observed returns; the latent instantaneous variance `v0` is filtered and updated **daily**.

To fit all four structural coefficients daily (slower, noisier):

```bash
python -m heston.calibrate_spx_returns \
  --start 2014-01-02 --end 2014-12-31 \
  --lookback 504 --refit-every 1 \
  --output heston/generated/spx_returns_heston_2014_daily_refit.csv
```

For an initial quick, lower-resolution test, shorten the period and use `--grid 32 --maxiter 15 --lookback 252`.

If automatic market-data retrieval fails, supply a downloaded CSV of **actual SPX prices** with `Date,Close` columns:

```bash
python -m heston.calibrate_spx_returns \
  --prices /path/to/actual_spx_closes.csv \
  --start 2014-01-02 --end 2014-12-31 --lookback 504 \
  --output heston/generated/spx_returns_heston_2014.csv
```

Include at least 504 earlier market sessions in your input CSV.

## Mathematical and statistical approach

The physical-measure model is

\[
d\log S_t=(\mu-\tfrac12v_t)\,dt+\sqrt{v_t}\,dW^S_t,\qquad
dv_t=\kappa(\theta-v_t)\,dt+\sigma\sqrt{v_t}\,dW^v_t,\qquad
d\langle W^S,W^v\rangle_t=\rho\,dt.
\]

The script fits a **finite-grid, discretised latent-variance likelihood** using the Euler Heston transition, including the correlation between the observed return shock and the subsequent variance innovation. The four structural parameters are optimised with bounded L-BFGS-B. `v0` is the filtered estimate of instantaneous variance after that trading day's return.

This is a genuine statistical calibration to an observed market series, **but not** an exact Heston transition density or an option-implied fit. The volatility grid, numerical truncation, assumed drift `--mu` (default annual 0.05), choice of rolling window and parameter bounds affect estimates. The underlying index alone may not strongly identify `rho` or `sigma`. No futures or ETF prices are substituted for SPX.

## Output and diagnostics

The first six columns are identical to the original source:

```csv
date,theta,kappa,sigma,rho,v0
```

Additional columns give refit dates, the data window, likelihood, optimiser status, constraint-bound warnings, real index closing price and data source. A neighbouring `.manifest.json` records the dataset source and SHA256, inputs, settings and counts of flagged estimates. The source 2013 file is never modified.

Use `--strict` if you want the program to **refuse** output whenever an optimiser result is flagged or approaches its constraints. Even unflagged fits are not guaranteed to be statistically well identified.

## Use with your existing Heston quantum-encoding code

```python
from heston.build_distributions import read_calibrations, build_distributions

cal = read_calibrations("heston/generated/spx_returns_heston_2014.csv")
batch = build_distributions(
    cal, n_qubits=8, cutoff=2.5,
    n_fft=16384, halfwidth=3.5,
    maturity=1.0, drift=0.05,
    verify_grid=True, fft_l1_tolerance=1e-4
)
print(batch.amplitudes.shape)
```

Note that the older 2013 FFT parameters may be inadequate for some of the newly estimated distributions. If the numerical PDF safeguards fail, adjust `halfwidth` and `n_fft` and inspect the problematic calibration; do not disable quality checks merely to make it run.

## Reproducibility and research caveat

The output CSV is only generated when **you actually run the pipeline and obtain observed index data**. A successful unit test does not demonstrate that the resulting estimates fit SPX returns well; examine the estimated bounds, residual distribution, stability under window/grid changes and predictive performance before using them as scientific evidence.

The physical-versus-risk-neutral distinction matters especially for comparing results with the existing 2013 source. For directly comparable option-implied calibrations, you would need historical SPX option quotes and a separate option-pricing fitting routine.

Upstream published 2013 CSV: [Empirical-Deep-Hedging](https://github.com/oskarimikkila/Empirical-Deep-Hedging).
