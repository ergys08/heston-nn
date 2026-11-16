"""Fast tests of cutoff-window and MPS complexity helpers."""
from __future__ import annotations
import unittest
import numpy as np
from .run_experiment import cdf, retained, radius, diagnostics

class CutoffTests(unittest.TestCase):
    def test_gaussian_radius(self):
        x=np.linspace(-10.,10.,20001)[:-1]
        p=np.exp(-.5*x*x)/np.sqrt(2*np.pi)
        xx,cc=cdf(x,p)
        r=radius(xx,cc,.99)
        self.assertAlmostEqual(r,2.575829,delta=.002)
        self.assertAlmostEqual(retained(xx,cc,r),.99,places=7)
    def test_basis_product(self):
        p=np.zeros(256);p[0]=1.
        d=diagnostics(p,[.995],3)
        self.assertEqual(d["max_exact_bond_dimension"],1)
        self.assertEqual(d["chi_for_995"],1)
        self.assertEqual(d["layers_for_995"],1)
    def test_ghz_rank(self):
        p=np.zeros(256);p[0]=p[-1]=.5
        d=diagnostics(p,[.995],2)
        self.assertEqual(d["max_exact_bond_dimension"],2)
        self.assertEqual(d["chi_for_995"],2)
if __name__=="__main__":unittest.main()
