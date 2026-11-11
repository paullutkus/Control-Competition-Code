"""Flexible-arm dynamics ported from latentctl/systems.py.

Source: Learning-Stabilizing-Latent-Feedback-Controllers,
commit 92231e2abbf0ed7e127d17d5b8c07c1e771d4bea.
Only the default, single-unstable-mode arm is retained. The inertia,
zero-order-hold linearization, and nonlinear momentum integrator are unchanged.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla


def zoh(Ac, Bc, dt):
    """Exact zero-order-hold discretization via the augmented matrix exponential."""
    n, p = Bc.shape
    Z = np.zeros((n + p, n + p))
    Z[:n, :n], Z[:n, n:] = Ac, Bc
    Phi = sla.expm(Z * dt)
    return Phi[:n, :n], Phi[:n, n:]


class FlexibleArm:
    """The paper's 20-section arm, in joint coordinates and normalized input units.

    State: (q_1, ..., q_20, qd_1, ..., qd_20), in radians and radians/second.
    Physical joint torque is u / u_scale, in N m. The feedback period is 0.05 s.
    """

    def __init__(self, substeps=50):
        if not isinstance(substeps, (int, np.integer)) or substeps < 1:
            raise ValueError("substeps must be a positive integer")
        n_sec, length, total_mass, gravity = 20, 1.0, 0.8, 9.81
        c0, c1, dt, u_max = 0.4, 0.02, 0.05, 40.0
        n_unstable, alpha_frac = 1, 0.5
        self.n_sec, self.dt, self.u_max = n_sec, dt, u_max
        ell = np.full(n_sec, length / n_sec)
        mass = np.full(n_sec, total_mass / n_sec)
        S = np.tril(np.ones((n_sec, n_sec)))                    # section angles phi = S q
        tail = np.cumsum(mass[::-1])[::-1] - mass
        w = gravity * ell * (0.5 * mass + tail)                 # gravity weights
        G = S.T @ np.diag(w) @ S                                # negative stiffness
        K0 = np.diag(np.linspace(1.0, 1.5, n_sec))
        mu = sla.eigh(G, K0, eigvals_only=True)                 # ascending buckling values
        k = n_sec - n_unstable - 1
        kappa = mu[k] + alpha_frac * (mu[k + 1] - mu[k])
        K_eff = kappa * K0 - G
        M = np.zeros((n_sec, n_sec))                            # upright inertia M(0)
        for i in range(n_sec):
            Jx = ell[:i] @ S[:i, :] + 0.5 * ell[i] * S[i, :]
            M += mass[i] * np.outer(Jx, Jx) + mass[i] * ell[i] ** 2 / 12.0 * np.outer(S[i], S[i])
        lever = np.tril(np.ones((n_sec, n_sec)), -1) * ell[None, :] + np.diag(0.5 * ell)   # a_ij
        Nphi = lever.T @ (mass[:, None] * lever)                # N_jk = sum_i m_i a_ij a_ik
        Mphi0 = Nphi + np.diag(mass * ell ** 2 / 12.0)          # M_phi(0), M(0) = S^T M_phi(0) S
        Minv = np.linalg.inv(M)
        Dmp = c0 * M + c1 * K_eff
        Z, I = np.zeros((n_sec, n_sec)), np.eye(n_sec)
        Ac = np.block([[Z, I], [-Minv @ K_eff, -Minv @ Dmp]])
        Bc = np.vstack([np.zeros((n_sec, 1)), (Minv @ np.ones(n_sec))[:, None]])
        A, B = zoh(Ac, Bc, dt)
        self.u_scale = float(np.linalg.norm(B))
        B = B / self.u_scale
        self.A, self.B, self.Ac, self.Bc = A, B, Ac, Bc
        self.M, self.Minv, self.K_eff, self.S, self.w, self.G, self.ell = M, Minv, K_eff, S, w, G, ell
        self.kappa, self.buckling = kappa, mu
        self.nu, self.V = sla.eigh(K_eff, M)                     # nu[0] < 0: buckling mode
        self.c_tip = ell @ S                                     # d tip_x / dq at q = 0
        self.Nphi, self.Mphi0, self.c0, self.c1 = Nphi, Mphi0, c0, c1
        self._Sinv, self._MinvST = np.linalg.inv(S), Minv @ S.T
        h = dt / substeps
        self._Eh = sla.expm(Ac * h)
        self._Ph = np.linalg.solve(Ac, self._Eh - np.eye(2 * n_sec))
        self._Wm = self._Ph / h
        self._substeps = substeps

    @property
    def n(self):
        return self.A.shape[0]

    @property
    def p(self):
        return self.B.shape[1]

    # ---- nonlinear plant (joint coordinates) ----------------------------------------
    def _dM_phi(self, phi):
        """M_phi(phi) - M_phi(0) = -2 N .* sin^2((phi_j - phi_k) / 2) for phi (k, n_sec)."""
        sh, ch = np.sin(0.5 * phi), np.cos(0.5 * phi)
        d = sh[:, :, None] * ch[:, None, :] - ch[:, :, None] * sh[:, None, :]      # sin((phi_j - phi_k) / 2)
        return -2.0 * self.Nphi * d * d

    def _step_exact(self, X, U, n_iter=3):
        """One ZOH period of the nonlinear plant.

        Momentum form: with m = (M(q) - M(0)) qd, the plant reads
            M(0) (qdd + c0 qd) = F_lin + rho - dm/dt,   rho = S^T (w .* (sin phi - phi) + dT/dphi)
                                                             - c0 (M(q) - M(0)) qd,
        F_lin the force of the linearization, T the kinetic energy.  Per substep h,
            x1 = E x0 + P_h B u + P_h (rho~0 + rho~1) / 2 - P_h / h (m~1 - m~0),
        E = e^(Ac h), P_h = int_0^h e^(Ac s) ds, y~ = (0, M(0)^-1 y): the variation-of-constants
        formula with rho and dm/dt constant over the substep.  A transient of the stiff
        (overdamped) bending modes thus enters the slow modes as the momentum change it causes
        (P_h / h ~ I there), not as its acceleration (an explicit residual weighs that by
        h |lambda_fast| ~ 300).  x1 is implicit: n_iter passes, each solving exactly for qd1
        given q1 (the m term is linear in qd1) and refreshing q1 and rho~1."""
        n, S, N = self.n_sec, self.S, self.Nphi
        Eh, Ph, W = self._Eh, self._Ph[:, n:], self._Wm[:, n:]
        MS = self._MinvST                                                             # M(0)^-1 S^T
        WMS = W[n:] @ MS

        def parts(X):
            phi, phid = X[:, :n] @ S.T, X[:, n:] @ S.T
            sn, cs = np.sin(phi), np.cos(phi)
            dM = self._dM_phi(phi)
            dMv = (dM @ phid[..., None])[..., 0]
            dT = -phid * (sn * ((cs * phid) @ N) - cs * ((sn * phid) @ N))         # dT/dphi
            return dM, (self.w * (sn - phi) + dT - self.c0 * dMv) @ MS.T, dMv @ MS.T  # dM_phi, rho~_v, m~_v

        Bu = (U / self.u_scale) @ self.Bc.T
        k, I = X.shape[0], np.eye(n)
        for _ in range(self._substeps):
            _, rho0, m0 = parts(X)
            lin = X @ Eh.T + Bu @ self._Ph.T
            b0 = lin + 0.5 * rho0 @ Ph.T + m0 @ W.T
            X1 = lin + rho0 @ Ph.T                                                    # predictor
            for _ in range(n_iter):
                dM, rho1, _ = parts(X1)
                b = b0 + 0.5 * rho1 @ Ph.T
                dMS = (dM.reshape(k * n, n) @ S).reshape(k, n, n)                     # dM_phi S (one GEMM)
                C = (dMS.transpose(0, 2, 1).reshape(k * n, n) @ WMS.T).reshape(k, n, n).transpose(0, 2, 1)
                qd1 = np.linalg.solve(I + C, b[:, n:, None])[..., 0]
                m1 = (dM @ (qd1 @ S.T)[..., None])[..., 0] @ MS.T
                X1 = np.concatenate([b[:, :n] - m1 @ W[:n].T, qd1], axis=1)
            X = X1
        return X

    def step(self, X, U, plant="nonlinear"):
        """Advance a state (40,) or batch (N, 40) with inputs (1,) or (N, 1).

        Inputs are held constant for dt seconds. This method does not saturate
        the input or stop on a fall (generate_data applies those rules).
        """
        if plant not in ("linear", "nonlinear"):
            raise ValueError("plant must be 'linear' or 'nonlinear'")
        X, U = np.asarray(X, dtype=float), np.asarray(U, dtype=float)
        single = X.ndim == 1
        X, U = np.atleast_2d(X), np.atleast_2d(U)
        if X.ndim != 2 or X.shape[1] != self.n or U.shape != (len(X), self.p):
            raise ValueError("expected states (N, 40) and inputs (N, 1)")
        out = X @ self.A.T + U @ self.B.T if plant == "linear" else self._step_exact(X, U)
        return out[0] if single else out

    # ---- physical quantities ----------------------------------------------------------
    def centerline(self, q):
        phi = self.S @ q
        return (np.concatenate([[0.0], np.cumsum(self.ell * np.sin(phi))]),
                np.concatenate([[0.0], np.cumsum(self.ell * np.cos(phi))]))

    def fallen(self, X):
        """Some section more than 90 degrees from upright."""
        return np.abs(X[..., :self.n_sec] @ self.S.T).max(axis=-1) > np.pi / 2
