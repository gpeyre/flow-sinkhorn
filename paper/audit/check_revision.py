#!/usr/bin/env python3
"""Numerical regression checks for formulas in the revised article, not proofs.

Run with a Python environment containing NumPy.
"""
import unittest

import numpy as np
from numpy.testing import assert_allclose


def kl(p, q):
    positive = p > 0
    return float(np.sum(p[positive]*np.log(p[positive]/q[positive])) - p.sum() + q.sum())


def logsumexp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x-m)))


def variation(v):
    return float(np.ptp(v) / 2)


class PaperRevisionChecks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20260927)
        self.n = 5
        undirected = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 3)]
        edges = [e for i, j in undirected for e in [(i, j), (j, i)]]
        self.i, self.j = np.array(edges).T
        self.w = np.repeat(rng.uniform(0.3, 1.1, len(undirected)), 2)
        self.z = np.exp(rng.normal(0, 0.3, len(edges)))
        self.gamma = 0.7
        self.theta = self.gamma / 2
        self.q = np.array([0.55, 0.45, 0, -0.3, -0.7])
        self.K = self.z * np.exp(-self.w / self.gamma)

    def psi2(self, v):
        return (v[self.i] + v[self.j]) / 2

    def psi1(self, U):
        log_out = np.array([
            logsumexp(np.log(self.z[self.i == a])
                      + (U[self.i == a] - self.w[self.i == a] / 2) / self.theta)
            for a in range(self.n)
        ])
        log_in = np.array([
            logsumexp(np.log(self.z[self.j == a])
                      + (-U[self.j == a] - self.w[self.j == a] / 2) / self.theta)
            for a in range(self.n)
        ])
        beta = self.q / 2 * np.exp(-(log_out + log_in) / 2)
        return self.theta * ((log_out - log_in) / 2 + np.arcsinh(beta))

    def recover(self, v, U):
        f = self.z * np.exp((-v[self.i] + U - self.w / 2) / self.theta)
        g = self.z * np.exp((v[self.j] - U - self.w / 2) / self.theta)
        return f, g

    def dual(self, v, U):
        f, g = self.recover(v, U)
        return self.q @ v + self.gamma * self.z.sum() - self.theta * (f.sum() + g.sum())

    def divergence(self, f):
        return np.bincount(self.j, weights=f, minlength=self.n) - np.bincount(
            self.i, weights=f, minlength=self.n)

    def test_bounded_mass_pinsker_without_equal_mass(self):
        rng = np.random.default_rng(10)
        for _ in range(300):
            p = rng.exponential(size=9) * rng.uniform(0.01, 4)
            q = rng.exponential(size=9) * rng.uniform(0.01, 4)
            X = max(p.sum(), q.sum())
            self.assertGreaterEqual(kl(p, q) + 1e-12, np.abs(p-q).sum() ** 2 / (2*X))
        p, q = np.array([0., 2., 0.]), np.array([1., .2, 3.])
        self.assertGreaterEqual(kl(p, q), np.abs(p-q).sum() ** 2 / (2*max(p.sum(), q.sum())))

    def test_projection_signs_half_cost_and_exact_ascent(self):
        v = np.linspace(-.2, .4, self.n)
        U = self.psi2(v)
        f, g = self.recover(v, U)
        assert_allclose(f, g, atol=2e-15)
        new_v = self.psi1(U)
        f1, g1 = self.recover(new_v, U)
        residual = -np.bincount(self.i, weights=f1, minlength=self.n) + np.bincount(
            self.j, weights=g1, minlength=self.n)
        assert_allclose(residual, self.q, atol=3e-15)
        assert_allclose(self.dual(new_v, U) - self.dual(v, U),
                        self.theta * (kl(f1, f) + kl(g1, g)), atol=3e-15)
        f2, g2 = self.recover(new_v, self.psi2(new_v))
        assert_allclose(f2, np.sqrt(f1*g1), atol=2e-15)
        assert_allclose(f2, g2, atol=2e-15)
        assert_allclose(self.dual(new_v, self.psi2(new_v)) - self.dual(new_v, U),
                        self.theta * (kl(f2, f1) + kl(g2, g1)), atol=3e-15)
        physical = self.w @ f2 + self.gamma * kl(f2, self.z)
        lifted = .5 * self.w @ (f2+g2) + self.theta * (kl(f2, self.z)+kl(g2, self.z))
        assert_allclose(lifted, physical, atol=3e-15)

    def test_primal_and_log_potential_updates_agree(self):
        v = np.linspace(-.4, .7, self.n)
        f, _ = self.recover(v, self.psi2(v))
        rows = np.bincount(self.i, weights=f, minlength=self.n)
        cols = np.bincount(self.j, weights=f, minlength=self.n)
        s = (np.sqrt(self.q**2+4*rows*cols)-self.q)/(2*rows)
        expected_v = v - self.gamma/2*np.log(s)
        assert_allclose(self.psi1(self.psi2(v)), expected_v, atol=2e-15)
        ap = np.array([self.gamma*logsumexp(np.log(self.K[self.i==a])+
                      v[self.j[self.i==a]]/self.gamma) for a in range(self.n)])
        am = np.array([self.gamma*logsumexp(np.log(self.K[self.j==a])-
                      v[self.i[self.j==a]]/self.gamma) for a in range(self.n)])
        stable_v = v/2 + (ap-am)/4 + self.gamma/2*np.arcsinh(
            self.q/2*np.exp(-(ap+am)/(2*self.gamma)))
        assert_allclose(stable_v, expected_v, atol=2e-15)
        fnew, _ = self.recover(stable_v, self.psi2(stable_v))
        assert_allclose(fnew, f*np.sqrt(s[self.i]/s[self.j]), atol=3e-15)

    def test_graph_order_and_variation_nonexpansiveness(self):
        rng = np.random.default_rng(3)
        for _ in range(100):
            v, w = rng.normal(size=(2, self.n))
            tv, tw = self.psi1(self.psi2(v)), self.psi1(self.psi2(w))
            self.assertLessEqual(variation(tv-tw), variation(v-w)+1e-12)
            assert_allclose(self.psi1(self.psi2(v+0.27)), tv+0.27, atol=2e-15)
            U = rng.normal(size=len(self.i))
            self.assertTrue(np.all(self.psi1(U+rng.uniform(size=len(U))) >= self.psi1(U)-1e-12))

    def test_optimum_reverse_edge_identity_and_log_bound(self):
        B = np.zeros((self.n, len(self.i)))
        B[self.j, np.arange(len(self.i))] = 1
        B[self.i, np.arange(len(self.i))] = -1
        v = np.zeros(self.n)
        for _ in range(100):
            f, _ = self.recover(v, self.psi2(v))
            gradient = self.divergence(f)-self.q
            if np.max(np.abs(gradient)) < 1e-10:
                break
            H = (B*f)@B.T/self.gamma
            step = np.r_[np.linalg.solve(H[:-1, :-1], gradient[:-1]), 0.]
            alpha = 1.
            old = self.dual(v, self.psi2(v))
            while self.dual(v-alpha*step, self.psi2(v-alpha*step)) < old-1e-14:
                alpha *= .5
                self.assertGreater(alpha, 1e-12)
            v -= alpha*step
        f, _ = self.recover(v, self.psi2(v))
        assert_allclose(self.divergence(f), self.q, atol=2e-7)
        logr = np.log(f/self.z)
        assert_allclose(logr[::2]+logr[1::2], -2*self.w[::2]/self.gamma, atol=2e-15)
        # Explicit feasible flow on edges 0->1->2->3->4 (orientation j->i).
        fbar = np.zeros(len(self.i))
        for a, amount in enumerate(np.cumsum(self.q)[:-1]):
            fbar[(self.i == a+1) & (self.j == a)] = amount
        assert_allclose(self.divergence(fbar), self.q, atol=3e-16)
        M = (self.w@fbar+self.gamma*kl(fbar,self.z))/self.w.min()
        H = abs(np.log(M))+np.abs(np.log(self.z)).max()+2*self.w.max()/self.gamma
        self.assertLessEqual(f.sum(), M)
        self.assertLessEqual(np.abs(logr).max(), H)

    def test_bias_envelope_for_arbitrary_reference_and_mass(self):
        rng = np.random.default_rng(12)
        for _ in range(100):
            x = rng.uniform(0, 8, size=7)
            z = np.exp(rng.normal(size=7))
            M = x.sum()*1.1
            B = z.sum()+M*np.log(max(1., M/z.min()))
            self.assertLessEqual(kl(x,z), B)

    def test_independent_block_gauges_do_not_bound_pairing(self):
        # A1=A2=[1]: N1=N2=R, while the joint kernel is {(c,-c)}.
        u = np.array([2., 2.])
        independent_radius = max(abs(u[0]-u[0]), abs(u[1]-u[1]))
        self.assertEqual(independent_radius, 0.)
        self.assertGreater(np.ones(2)@u, 0.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
