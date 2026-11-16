"""Numerical checks of cutoff-window and standardised tail-shape utilities."""
from __future__ import annotations
import unittest
import numpy as np
from .run_experiment import (cdf, retained, radius, diagnostics,
                             full_moments, tail_z, z_radius, bin_probs)


class CutoffTests(unittest.TestCase):
    def _normal(self, mu=0., sd=1.):
        x=np.linspace(-15.,15.,30001)[:-1]
        p=np.exp(-.5*((x-mu)/sd)**2)/(sd*np.sqrt(2*np.pi))
        return x,p

    def test_gaussian_radius(self):
        x,p=self._normal()
        xx,cc=cdf(x,p)
        r=radius(xx,cc,.99)
        self.assertAlmostEqual(r,2.575829,delta=.002)
        self.assertAlmostEqual(retained(xx,cc,r),.99,places=7)

    def test_gaussian_excess_kurtosis_and_3sigma_mass(self):
        x,p=self._normal()
        xx,cc=cdf(x,p)
        mu,sd,var,skew,kurt=full_moments(x,p)
        self.assertAlmostEqual(mu,0.,delta=2e-4)
        self.assertAlmostEqual(sd,1.,delta=1e-4)
        self.assertAlmostEqual(skew,0.,delta=1e-3)
        self.assertAlmostEqual(kurt,0.,delta=1e-3)
        self.assertAlmostEqual(tail_z(xx,cc,mu,sd,3),.002699796,delta=2e-5)

    def test_scale_invariant_shape_and_standardised_quantiles(self):
        z,pz=self._normal()
        x=4. + 2.5*z
        p=pz/2.5
        xx,cc=cdf(x,p)
        mu,sd,_,_,kurt=full_moments(x,p)
        self.assertAlmostEqual(mu,4.,delta=2e-3)
        self.assertAlmostEqual(sd,2.5,delta=2e-3)
        self.assertAlmostEqual(kurt,0.,delta=1e-3)
        self.assertAlmostEqual(tail_z(xx,cc,mu,sd,3),.002699796,delta=2e-5)
        self.assertAlmostEqual(z_radius(xx,cc,mu,sd,.99),2.575829,delta=.003)
        a,retain_a=bin_probs(xx,cc,mu-3*sd,mu+3*sd,8)
        zz,zc=cdf(z,pz)
        b,retain_b=bin_probs(zz,zc,-3,3,8)
        np.testing.assert_allclose(a,b,rtol=2e-3,atol=2e-6)
        self.assertAlmostEqual(retain_a,retain_b,delta=1e-5)

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


if __name__=="__main__":
    unittest.main()
