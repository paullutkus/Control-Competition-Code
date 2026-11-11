"""Checks for the mechanical model, feedback, and trajectory file semantics."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import solve_ivp

from flexible_arm import FlexibleArm, generate_data, load_controller


class FlexibleArmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arm = FlexibleArm()

    def test_open_and_closed_loop_spectra(self):
        arm = self.arm
        self.assertEqual(arm.A.shape, (40, 40))
        self.assertEqual(arm.B.shape, (40, 1))
        self.assertAlmostEqual(np.linalg.norm(arm.B), 1.0)
        self.assertEqual(np.count_nonzero(abs(np.linalg.eigvals(arm.A)) > 1), 1)
        stable = max(abs(np.linalg.eigvals(arm.A + arm.B @ load_controller("stable"))))
        unstable = max(abs(np.linalg.eigvals(arm.A + arm.B @ load_controller("unstable"))))
        self.assertAlmostEqual(stable, 0.9549194915, places=7)
        self.assertAlmostEqual(unstable, 1.1293703152, places=7)

    def test_nonlinear_step_has_correct_linearization(self):
        arm = self.arm
        delta = 1e-7
        perturbations = np.vstack([delta * np.eye(40), -delta * np.eye(40)])
        states = arm.step(perturbations, np.zeros((80, 1)))
        jacobian = ((states[:40] - states[40:]) / (2 * delta)).T
        np.testing.assert_allclose(jacobian, arm.A, atol=1e-8, rtol=1e-8)
        inputs = arm.step(np.zeros((2, 40)), np.array([[delta], [-delta]]))
        np.testing.assert_allclose((inputs[0] - inputs[1]) / (2 * delta), arm.B[:, 0],
                                   atol=1e-8, rtol=1e-8)
        np.testing.assert_array_equal(arm.step(np.zeros(40), np.zeros(1)), np.zeros(40))

    def test_nonlinear_step_against_mechanical_equations(self):
        arm = self.arm
        # Independently assemble the inertia and centrifugal force from the
        # section geometry, then integrate the second-order equations with Radau.
        ell, mass = np.full(20, 0.05), np.full(20, 0.04)
        lever = np.tril(np.ones((20, 20)), -1) * ell[None, :] + np.diag(ell / 2)
        N = lever.T @ (mass[:, None] * lever)
        inertia = np.diag(mass * ell ** 2 / 12)
        K0 = np.diag(np.linspace(1, 1.5, 20))
        x0 = 0.01 * np.random.default_rng(17).standard_normal(40)
        u = np.array([0.02])

        def rhs(t, x):
            q, velocity = x[:20], x[20:]
            phi, phid = arm.S @ q, arm.S @ velocity
            differences = phi[:, None] - phi[None, :]
            M = arm.S.T @ (N * np.cos(differences) + inertia) @ arm.S
            centrifugal = arm.S.T @ np.sum(N * np.sin(differences) * phid[None, :] ** 2, axis=1)
            force = (np.ones(20) * u[0] / arm.u_scale - centrifugal
                     - (arm.c0 * M + arm.c1 * arm.K_eff) @ velocity
                     - arm.kappa * K0 @ q + arm.S.T @ (arm.w * np.sin(phi)))
            return np.concatenate([velocity, np.linalg.solve(M, force)])

        reference = solve_ivp(rhs, (0, arm.dt), x0, method="Radau", rtol=1e-10, atol=1e-12)
        self.assertTrue(reference.success)
        np.testing.assert_allclose(arm.step(x0, u), reference.y[:, -1], atol=2e-6, rtol=0)

    def test_paired_nonlinear_rollouts_and_padding(self):
        args = dict(trajectories=4, steps=160, seed=0, arm=self.arm)
        stable = generate_data("stable", **args)
        unstable = generate_data("unstable", **args)
        np.testing.assert_array_equal(stable["X"][:, 0], unstable["X"][:, 0])
        self.assertTrue(np.all(stable["status"] == "completed"))
        ratio = np.linalg.norm(stable["X"][:, -1], axis=1) / np.linalg.norm(stable["X"][:, 0], axis=1)
        self.assertTrue(np.all(ratio < 0.01))
        self.assertTrue(np.all(unstable["status"] == "fallen"))
        for i, length in enumerate(unstable["lengths"]):
            self.assertGreater(length, 0)
            self.assertTrue(np.all(np.isfinite(unstable["X"][i, :length + 1])))
            self.assertTrue(np.all(np.isfinite(unstable["U"][i, :length])))
            self.assertTrue(np.all(np.isnan(unstable["X"][i, length + 1:])))
            self.assertTrue(np.all(np.isnan(unstable["U"][i, length:])))
            self.assertTrue(self.arm.fallen(unstable["X"][i, length]))
            self.assertFalse(np.any(self.arm.fallen(unstable["X"][i, :length])))

    def test_saturation_units_and_linear_transition(self):
        arm = self.arm
        x0 = 1000 * load_controller("stable")
        data = generate_data("stable", plant="linear", initial_states=x0, steps=1, arm=arm)
        self.assertAlmostEqual(abs(data["torque"][0, 0, 0]), 40)
        np.testing.assert_allclose(data["U"], data["torque"] * arm.u_scale)
        np.testing.assert_allclose(data["X"][:, 1], x0 @ arm.A.T + data["U"][:, 0] @ arm.B.T)
        self.assertEqual(data["status"][0], "completed")

    def test_termination_at_initial_state(self):
        x0 = np.zeros((2, 40))
        x0[0, 0] = np.pi
        data = generate_data(initial_states=x0, steps=2, arm=self.arm)
        np.testing.assert_array_equal(data["lengths"], [0, 2])
        np.testing.assert_array_equal(data["status"], ["fallen", "completed"])
        self.assertTrue(np.all(np.isnan(data["X"][0, 1:])))
        np.testing.assert_array_equal(data["X"][1], np.zeros((3, 40)))

    def test_failed_step_does_not_insert_zero_states(self):
        def failed_member(states, inputs, plant):
            result = states.copy()
            result[0] = np.nan
            return result

        x0 = np.full((2, 40), 0.001)
        with patch.object(self.arm, "step", side_effect=failed_member):
            data = generate_data(initial_states=x0, steps=1, arm=self.arm)
        np.testing.assert_array_equal(data["lengths"], [0, 1])
        np.testing.assert_array_equal(data["status"], ["numerical_failure", "completed"])
        self.assertTrue(np.all(np.isnan(data["X"][0, 1:])))
        self.assertTrue(np.all(np.isnan(data["U"][0])))
        np.testing.assert_array_equal(data["X"][1, 1], x0[1])

    def test_command_line_writes_paired_files_without_pickle(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as output:
            subprocess.run([sys.executable, str(root / "generate_data.py"),
                            "--plant", "linear", "--trajectories", "2", "--steps", "3",
                            "--output", output], check=True, capture_output=True, text=True)
            with np.load(Path(output) / "stable.npz", allow_pickle=False) as stable, \
                    np.load(Path(output) / "unstable.npz", allow_pickle=False) as unstable:
                np.testing.assert_array_equal(stable["X"][:, 0], unstable["X"][:, 0])
                self.assertEqual(stable["X"].shape, (2, 4, 40))
                np.testing.assert_allclose(stable["time"], [0, 0.05, 0.1, 0.15])
                for key in stable.files:
                    self.assertNotEqual(stable[key].dtype, np.dtype("object"))


if __name__ == "__main__":
    unittest.main()
