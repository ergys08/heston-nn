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
