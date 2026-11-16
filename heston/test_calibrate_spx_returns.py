"""Smoke tests for real-SPX Heston return calibration. No network required."""
from __future__ import annotations
import unittest
import numpy as np
from .calibrate_spx_returns import HestonReturnFilter, fit_window, calibrate_dates, read_price_csv


class TestSPXReturnCalibration(unittest.TestCase):
    def test_csv_validation(self):
        with self.assertRaises(ValueError):
            read_price_csv("Date,Price\n2020-01-02,100", source="test")
        rows = "Date,Close\n" + "".join(
            f"2020-{1+i//28:02d}-{1+i%28:02d},{100+i}\n" for i in range(70)
        )
        dates, close = read_price_csv(rows, source="test")
        self.assertEqual(len(dates), 70)
        self.assertEqual(float(close[-1]), 169.)

    def test_filter_normalization(self):
        model = HestonReturnFilter(grid_size=32)
        p = np.array([3., .025, .65, -.6])
        returns = np.random.default_rng(4).normal(0, .01, 100)
        ll, last, trajectory = model.filter(returns, p, save_path=True)
        self.assertTrue(np.isfinite(ll))
        self.assertEqual(len(trajectory), len(returns))
        self.assertAlmostEqual(last.sum(), 1., places=10)
        self.assertTrue(np.all(trajectory > 0))

    def test_optimisation(self):
        model = HestonReturnFilter(grid_size=32)
        returns = np.random.default_rng(5).normal(0, .009, 160)
        guess = np.array([3., .05, .6, -.65])
        ll0, _, _ = model.filter(returns, guess)
        fitted, ll1, _, _, _ = fit_window(model, returns, previous=guess, maxiter=8)
        self.assertGreaterEqual(ll1, ll0 - 1e-7)
        self.assertTrue(np.all(np.isfinite(fitted)))

    def test_no_future_lookahead(self):
        dates = np.busday_offset("2012-01-02", np.arange(520)).astype("datetime64[D]")
        rng = np.random.default_rng(6)
        closes = 100 * np.exp(np.r_[0., np.cumsum(rng.normal(0, .009, 519))])
        options = dict(start=str(dates[500]), end=str(dates[502]),
                       lookback=252, refit_every=21, grid=28, mu=.05, maxiter=2,
                       max_variance=1., source_meta={"provider":"test"})
        original = calibrate_dates(dates, closes, **options)
        edited = closes.copy()
        edited[503:] *= 10
        new = calibrate_dates(dates, edited, **options)
        self.assertEqual(len(original), 3)
        self.assertEqual([r["refit"] for r in original], [True, False, False])
        for a,b in zip(original,new):
            for k in ("kappa","theta","sigma","rho","v0"):
                self.assertAlmostEqual(a[k], b[k], places=9)


if __name__ == "__main__":
    unittest.main()
