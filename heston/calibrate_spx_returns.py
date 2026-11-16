"""Rolling Heston (physical-measure) calibration to REAL S&P 500 index returns.

Use as a module: python -m heston.calibrate_spx_returns ...

The stochastic-variance state is latent. We approximate the Euler-discretised
Heston model with a finite-grid hidden Markov filter. The conditional variance
transition respects the correlation between return and variance innovations.
This is a genuine statistical fit to observed index closes; it is NOT an
option-implied/risk-neutral calibration and must not be presented as one.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import gamma as gamma_dist

LOG = logging.getLogger("heston.spx_returns")
DT = 1.0 / 252.0
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "heston" / "generated" / "spx_return_calibrations.csv"
PARAM_NAMES = ("kappa", "theta", "sigma", "rho")
BOUNDS = ((0.12, 20.0), (0.001, 0.4), (0.025, 2.5), (-0.97, 0.97))


def read_price_csv(content: str, *, source: str) -> tuple[np.ndarray, np.ndarray]:
    """Accept Date,Close (Stooq/Yahoo) or date,close. Never interpolate prices."""
    reader = csv.DictReader(io.StringIO(content.lstrip("\ufeff")))
    if not reader.fieldnames:
        raise ValueError(f"{source}: empty/non-CSV market data")
    fields = {k.strip().lower(): k for k in reader.fieldnames}
    if not {"date", "close"}.issubset(fields):
        raise ValueError(f"{source}: expected columns Date and Close; got {reader.fieldnames}")
    rows = []
    for row in reader:
        try:
            date = np.datetime64(row[fields["date"]][:10], "D")
            close = float(row[fields["close"]])
        except (ValueError, TypeError, KeyError):
            continue
        if np.isfinite(close) and close > 0 and not np.isnat(date):
            rows.append((date, close))
    if not rows:
        raise ValueError(f"{source}: no valid positive-price observations")
    rows.sort(key=lambda x: x[0])
    dates, closes = zip(*rows)
    dates = np.asarray(dates, dtype="datetime64[D]")
    if len(np.unique(dates)) != len(dates):
        raise ValueError(f"{source}: duplicate trading dates, refuse to silently select one")
    if len(dates) < 50:
        raise ValueError(f"{source}: fewer than 50 usable observations")
    return dates, np.asarray(closes, dtype=np.float64)


def obtain_prices(provider: str, file: Path | None, history_start: str, end: str):
    """Return dates, closes, and source metadata. Only actual market data."""
    if file is not None:
        raw = file.read_bytes()
        dates, closes = read_price_csv(raw.decode("utf-8-sig"), source=str(file))
        return dates, closes, {"provider": "supplied-csv", "source": str(file),
                                "sha256": hashlib.sha256(raw).hexdigest()}

    errors = []
    if provider in ("auto", "stooq"):
        params = urlencode({"s": "^spx", "i": "d", "d1": history_start.replace("-", ""),
                            "d2": end.replace("-", "")})
        url = "https://stooq.com/q/d/l/?" + params
        try:
            raw = urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0 (research; heston-nn)"}), timeout=35).read()
            dates, closes = read_price_csv(raw.decode("utf-8-sig"), source="Stooq SPX")
            return dates, closes, {"provider": "stooq", "source": url,
                                    "sha256": hashlib.sha256(raw).hexdigest()}
        except Exception as e:
            errors.append(f"Stooq: {e}")
            if provider == "stooq":
                raise RuntimeError("Could not download REAL SPX prices: " + "; ".join(errors)) from e
    if provider in ("auto", "yfinance"):
        try:
            import yfinance as yf  # optional dependency
            # yf end is exclusive. Include the requested final trading date.
            end_exclusive = str(np.datetime64(end) + np.timedelta64(1, "D"))
            df = yf.download("^GSPC", start=history_start, end=end_exclusive,
                             auto_adjust=False, progress=False, threads=False)
            if df.empty:
                raise ValueError("Yahoo returned no SPX prices")
            series = df["Close"]
            if getattr(series, "ndim", 1) > 1:
                series = series.iloc[:, 0]
            raw = "Date,Close\n" + "".join(f"{i.date()},{float(x)}\n" for i, x in series.items())
            dates, closes = read_price_csv(raw, source="Yahoo ^GSPC")
            return dates, closes, {"provider": "yfinance", "source": "Yahoo Finance ^GSPC",
                                    "sha256": hashlib.sha256(raw.encode()).hexdigest()}
        except Exception as e:
            errors.append(f"Yahoo/yfinance: {e}")
    raise RuntimeError("Failed to obtain actual SPX observations: " + "; ".join(errors) +
                       ". Provide --prices /path/to/real_SPX_daily.csv if needed.")


class HestonReturnFilter:
    """Approximate Heston Euler HMM with latent variance on a fixed grid.

    r_t | v_t ~ N((mu-v_t/2)*dt, v_t*dt).
    v_{t+1} | (v_t,r_t) ~ N(m, s^2), where
      m = v_t + kappa*(theta-v_t)*dt + sigma*rho*(r_t-(mu-v_t/2)*dt),
      s^2 = sigma^2*v_t*dt*(1-rho^2).
    Variance positivity is approximated by clipping the Euler transition to
    the first grid cell. This is NOT exact CIR sampling/likelihood.
    """

    def __init__(self, grid_size: int = 64, max_variance: float = 1.0, mu: float = 0.05):
        if grid_size < 24 or max_variance <= 0:
            raise ValueError("Need grid_size >= 24 and max_variance > 0")
        self.v = np.geomspace(1e-6, max_variance, grid_size)
        self.edges = np.r_[0.0, np.sqrt(self.v[:-1] * self.v[1:]), np.inf]
        self.widths = np.diff(np.r_[self.edges[:-1], max_variance * 1.5])
        self.mu = float(mu)
        self.base_logv = np.log(2 * np.pi * self.v * DT)
        self.sqrtvdt = np.sqrt(self.v * DT)

    def initial_prior(self, params: np.ndarray):
        kappa, theta, sigma, _ = params
        shape = max(0.06, 2 * kappa * theta / sigma**2)
        scale = max(1e-12, sigma**2 / (2 * kappa))
        # CDF differences integrate gamma stationary distribution over bins.
        cdf = gamma_dist.cdf(self.edges, a=shape, scale=scale)
        weights = np.maximum(np.diff(cdf), 0)
        total = weights.sum()
        if not np.isfinite(total) or total < 1e-200:
            weights = np.exp(-0.5 * ((np.log(self.v) - np.log(theta)) / 0.8) ** 2)
            total = weights.sum()
        return weights / total

    def step(self, prior: np.ndarray, r: float, params: np.ndarray):
        kappa, theta, sigma, rho = params
        eps = r - (self.mu - 0.5 * self.v) * DT
        ll_vec = -0.5 * (self.base_logv + (eps / self.sqrtvdt) ** 2)
        maxll = float(np.max(ll_vec))
        w = prior * np.exp(ll_vec - maxll)
        evidence = float(w.sum())
        if evidence <= 0 or not np.isfinite(evidence):
            return np.full_like(prior, 1 / len(prior)), -1e10
        posterior = w / evidence
        means = self.v + kappa * (theta - self.v) * DT + sigma * rho * eps
        std = sigma * self.sqrtvdt * np.sqrt(max(1 - rho * rho, 1e-8))
        # Calculate bin probabilities; folding negative v into first bin and
        # above-range v into last bin preserves probability mass.
        z = (self.edges[1:-1][None, :] - means[:, None]) / std[:, None]
        cdf = ndtr(z)
        trans = np.diff(np.concatenate([np.zeros((len(self.v), 1)), cdf,
                                        np.ones((len(self.v), 1))], axis=1), axis=1)
        next_prior = posterior @ np.maximum(trans, 0)
        next_prior = np.maximum(next_prior, 0)
        next_prior /= next_prior.sum()
        return next_prior, float(maxll + np.log(evidence))

    def filter(self, returns: np.ndarray, params: np.ndarray, save_path: bool = False):
        prior = self.initial_prior(params)
        ll = 0.0
        conditional_variances = []
        for r in np.asarray(returns, float):
            prior, contrib = self.step(prior, float(r), params)
            ll += contrib
            if save_path:
                conditional_variances.append(float(prior @ self.v))
        return ll, prior, np.asarray(conditional_variances)


def fit_window(filter_model: HestonReturnFilter, returns: np.ndarray,
               *, previous: np.ndarray | None = None, maxiter: int = 60):
    """Likelihood fit on past *actual returns*. Never uses future prices."""
    annual_var = max(0.002, min(0.3, float(np.mean(returns**2) / DT)))
    seeds = [np.array([3.0, annual_var, 0.6, -0.65])]
    if previous is not None:
        seeds.insert(0, previous.copy())
    best = None
    for seed in seeds:
        def objective(x):
            ll, _, _ = filter_model.filter(returns, x)
            return -ll / len(returns)
        result = minimize(objective, seed, method="L-BFGS-B", bounds=BOUNDS,
                          options={"maxiter": maxiter, "ftol": 1e-9, "maxls": 16})
        if best is None or result.fun < best.fun:
            best = result
    params = np.asarray(best.x, float)
    ll, end_prior, _ = filter_model.filter(returns, params)
    boundary = [name for name, x, (lo, hi) in zip(PARAM_NAMES, params, BOUNDS)
                if x - lo < 0.002 * (hi - lo) or hi - x < 0.002 * (hi - lo)]
    return params, ll, end_prior, bool(best.success), boundary


def calibrate_dates(dates: np.ndarray, close: np.ndarray, *, start: str, end: str,
                    lookback: int, refit_every: int, grid: int, mu: float,
                    maxiter: int, max_variance: float, source_meta: dict):
    if lookback < 126 or refit_every < 1:
        raise ValueError("Use lookback >=126 daily returns and refit_every >= 1")
    if not (dates[0] < np.datetime64(start)):
        raise ValueError("Need historic prices BEFORE --start for calibration history")
    returns = np.diff(np.log(close))
    r_dates = dates[1:]
    selected = np.where((r_dates >= np.datetime64(start)) & (r_dates <= np.datetime64(end)))[0]
    if not len(selected):
        raise ValueError("No daily market observations in requested date range")
    if selected[0] < lookback - 1:
        raise ValueError(f"Insufficient SPX price history: need {lookback} returns before first output")
    fmodel = HestonReturnFilter(grid_size=grid, max_variance=max_variance, mu=mu)
    rows, last_p, prior, last_ll = [], None, None, None
    for j, i in enumerate(selected):
        refit = j == 0 or (j % refit_every == 0)
        if refit:
            sample = returns[i + 1 - lookback:i + 1]
            p, last_ll, prior, ok, bound = fit_window(fmodel, sample, previous=last_p, maxiter=maxiter)
            last_p = p
            last_ok, last_bound = ok, bound
            sample_start = str(r_dates[i + 1 - lookback]); fit_start = sample_start
            fit_end = str(r_dates[i])
            LOG.info("%s refit: kappa=%.4g theta=%.4g sigma=%.4g rho=%.4g v0=%.4g logL=%.2f success=%s bounds=%s",
                     r_dates[i], *p, prior @ fmodel.v, last_ll, ok, bound)
        else:
            prior, _ = fmodel.step(prior, returns[i], last_p)
        # Heston v0 is the forecast instantaneous variance immediately after
        # close on this date. Daily kappa/theta/sigma/rho are rolling estimates.
        v0 = float(prior @ fmodel.v)
        record = {"date": str(r_dates[i]),
                  "theta": float(last_p[1]), "kappa": float(last_p[0]),
                  "sigma": float(last_p[2]), "rho": float(last_p[3]), "v0": v0,
                  "refit": bool(refit), "n_returns": lookback,
                  "fit_window_start": fit_start, "fit_window_end": fit_end,
                  "fit_loglik": float(last_ll), "optimiser_success": bool(last_ok),
                  "at_bound": ";".join(last_bound), "spot_close": float(close[i + 1]),
                  "source": source_meta["provider"], "method": "physical_returns_euler_grid_mle"}
        rows.append(record)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--start", default="2014-01-02", help="First output trading date")
    p.add_argument("--end", default="2014-12-31", help="Last output trading date")
    p.add_argument("--history-start", default="2010-01-01", help="Earliest requested source price")
    p.add_argument("--provider", choices=("auto", "stooq", "yfinance"), default="auto")
    p.add_argument("--prices", type=Path, default=None, help="Local CSV with REAL Date,Close SPX prices, instead of downloading")
    p.add_argument("--lookback", type=int, default=504, help="Past returns used at each refit")
    p.add_argument("--refit-every", type=int, default=21, help="Refit four structural Heston parameters every N trading days; infer v0 daily")
    p.add_argument("--grid", type=int, default=64, help="Number of variance-grid points")
    p.add_argument("--max-variance", type=float, default=1.0)
    p.add_argument("--maxiter", type=int, default=60, help="Maximum optimiser iterations per initialisation")
    p.add_argument("--mu", type=float, default=0.05, help="Assumed physical annual SPX drift (nuisance; NOT risk-free rate)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--strict", action="store_true", help="Fail rather than output flagged at-bound or optimiser-failed estimates")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.start > args.end:
        p.error("--start must not exceed --end")
    dates, close, meta = obtain_prices(args.provider, args.prices, args.history_start, args.end)
    rows = calibrate_dates(dates, close, start=args.start, end=args.end,
                           lookback=args.lookback, refit_every=args.refit_every,
                           grid=args.grid, mu=args.mu, maxiter=args.maxiter,
                           max_variance=args.max_variance, source_meta=meta)
    problems = [r for r in rows if not r["optimiser_success"] or r["at_bound"]]
    if args.strict and problems:
        raise RuntimeError(f"{len(problems)} estimates flagged: no output written in --strict mode")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    manifest = {"instrument": "S&P 500 price index (SPX / ^GSPC)",
                "source": meta, "source_first_date": str(dates[0]), "source_last_date": str(dates[-1]),
                "model": "Heston under P, approximate Euler latent-variance grid likelihood; NOT option-implied Q",
                "requested_range": [args.start, args.end], "n_rows": len(rows),
                "lookback": args.lookback, "refit_every": args.refit_every,
                "mu": args.mu, "grid": args.grid, "max_variance": args.max_variance,
                "bounds": {k: b for k, b in zip(PARAM_NAMES, BOUNDS)},
                "n_rows_flagged": len(problems),
                "n_refits": sum(r["refit"] for r in rows)}
    manifest_file = args.output.with_suffix(".manifest.json")
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {len(rows)} actual-market-return-based Heston calibration rows to {args.output}")
    print(f"Saved provenance/quality report: {manifest_file}")
    if problems:
        print(f"WARNING: {len(problems)} rows have optimiser/boundary warnings. Review quality columns.")
    print("Physical-measure return fit; cannot be compared directly with 2013 option-implied risk-neutral parameters.")


if __name__ == "__main__":
    main()
