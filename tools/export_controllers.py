"""Re-export the trusted local paper checkpoints using only NumPy and SciPy.

This is optional: controller_data.npz is shipped with the code. Pickle is used
only to read the original repository's checkpoints; load only trusted files.
"""

import argparse
import hashlib
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import scipy.linalg as sla

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flexible_arm import FlexibleArm  # noqa: E402


def value_and_jacobian(layers):
    """Value and column-convention Jacobian of a tanh MLP at zero (float64)."""
    value = np.zeros(layers[0][0].shape[0])
    jacobian = np.eye(len(value))
    for i, (weight, bias) in enumerate(layers):
        weight, bias = np.asarray(weight, dtype=float), np.asarray(bias, dtype=float)
        value = value @ weight + bias
        jacobian = jacobian @ weight
        if i < len(layers) - 1:
            value = np.tanh(value)
            jacobian = jacobian * (1.0 - value ** 2)
    return value, jacobian.T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path.home() / "Documents/Learning-Stabilizing-Latent-Feedback-Controllers")
    parser.add_argument("--output", type=Path, default=ROOT / "flexible_arm/controller_data.npz")
    args = parser.parse_args()

    files = {"stable": "mlp__nonlinear__s0__sr__hT.pkl",
             "unstable": "mlp__nonlinear__s0__Lf__sr__hT.pkl"}
    exported = {"source_revision": subprocess.check_output(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()}
    arm = FlexibleArm()
    for name, filename in files.items():
        path = args.source / "results/paper/motivating_example" / filename
        raw = path.read_bytes()
        checkpoint = pickle.loads(raw)
        record = checkpoint["recs"][40]
        parameters = record["params"]
        sx, sz = record["scales"]
        # Anchoring subtracts the value at zero without changing the Jacobian.
        E = (sz / sx) * value_and_jacobian(parameters["E"])[1]
        L = value_and_jacobian(parameters["f"])[1]
        H = value_and_jacobian(parameters["g"])[0].reshape(4, 1)
        P = sla.solve_discrete_are(L, H, np.eye(4), np.eye(1))
        K = -np.linalg.solve(np.eye(1) + H.T @ P @ H, H.T @ P @ L)
        radius = float(np.max(np.abs(np.linalg.eigvals(arm.A + arm.B @ K @ E))))
        if (radius < 1.0) != (name == "stable"):
            raise RuntimeError(f"unexpected {name} closed-loop spectral radius: {radius}")
        for key, value in dict(E=E, L=L, H=H, K=K).items():
            exported[f"{name}_{key}"] = value
        exported[f"{name}_checkpoint"] = filename
        exported[f"{name}_sha256"] = hashlib.sha256(raw).hexdigest()
        print(f"{name}: rho(A + B K E) = {radius:.10f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **exported)


if __name__ == "__main__":
    main()
