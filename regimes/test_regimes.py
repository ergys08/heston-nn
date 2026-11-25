"""Small numerical and structural checks; run with unittest."""
import itertools
import unittest
import numpy as np

from .analysis import spectra, effective_rank, best_order, analyse, fidelity_curve, reorder_state
from .states import Coordinates, moments, make_state
from .run_experiment import grids, sample_models, AXES, REFERENCE


class TestRegimes(unittest.TestCase):
    def test_coordinates(self):
        q = Coordinates(3., .2, 2., .04, -.7)
        T = 2.
        p = q.parameters(T)
        np.testing.assert_allclose([p.kappa*T, p.xi**2/(2*p.kappa*p.theta),
                                    p.v0/p.theta, p.theta*T, p.rho],
                                   [q.a,q.b,q.c,q.d,q.rho])
        mean, std, EA, VA = moments(p, T)
        self.assertAlmostEqual(mean, -.5*EA)
        self.assertGreater(std, 0)
        self.assertGreater(VA, 0)

    def test_product_and_ghz(self):
        product = np.ones(16)/4
        ghz = np.zeros(16)
        ghz[[0,-1]] = 1/np.sqrt(2)
        for state, expected in [(product,1),(ghz,2)]:
            w, curves, rows = analyse(state, [1e-6])
            self.assertEqual(len(w), 7)
            self.assertEqual(rows[0]["natural_cut_bound"], expected)
            self.assertEqual(rows[0]["natural_tt_cap"], expected)
            self.assertEqual(rows[0]["best_cut_bound"], expected)

    def test_crossed_bell_pairs(self):
        # Bell pairs (q0,q2) and (q1,q3): natural cut crosses both.
        psi = np.zeros(16)
        psi[[0,5,10,15]] = .5
        _, _, rows = analyse(psi, [1e-6])
        row = rows[0]
        self.assertEqual(row["natural_cut_bound"], 4)
        self.assertEqual(row["best_cut_bound"], 2)
        self.assertEqual(row["reordered_tt_cap"], 2)

    def test_dp_against_bruteforce(self):
        rng = np.random.default_rng(123)
        n = 5
        psi = rng.random(1<<n)
        w = spectra(psi)
        full = (1<<n)-1
        eps = .02
        def cost(order):
            mask, ranks = 0, []
            for q in order[:-1]:
                mask |= 1<<q
                ranks.append(effective_rank(w[min(mask,full^mask)],eps))
            return max(ranks), order
        self.assertEqual(best_order(w,n,eps), min(map(cost,itertools.permutations(range(n)))))

    def test_reference_and_grid(self):
        g = grids(3,REFERENCE)
        cases = list(sample_models(g,REFERENCE,AXES,"pairwise"))
        self.assertEqual(len(cases),len(set(cases)))
        self.assertIn(REFERENCE,cases)
        self.assertIn(1.,g["b"])
        self.assertIn(0.,g["rho"])
        self.assertTrue(all(sum(x != y for x,y in zip(c,REFERENCE)) <= 2 for c in cases))

    def test_fft_reference(self):
        p, diag = make_state(Coordinates(*REFERENCE), n_qubits=6,
                             n_fft=4096,max_fft=262144)
        self.assertEqual(len(p),64)
        self.assertAlmostEqual(float(p.sum()),1.)
        self.assertTrue(np.all(p>=0))
        self.assertGreater(diag["retained_mass"],.9)
        self.assertLessEqual(diag["resolution_infidelity"],1e-8)
        self.assertLessEqual(diag["domain_infidelity"],1e-8)


if __name__ == "__main__":
    unittest.main()
