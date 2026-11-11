"""Feedback gains exported from the two models of the motivating example."""

from pathlib import Path

import numpy as np


def load_controller(name):
    """Return F (1, 40) for the normalized, unsaturated input u = F x.

    'stable' is the backward-loss model; 'unstable' is the forward-loss model.
    Both use the paper's linear feedback K J_E(0), with a four-state latent LQR.
    The nonlinear encoder is not evaluated during feedback.
    """
    if name not in ("stable", "unstable"):
        raise ValueError("controller must be 'stable' or 'unstable'")
    with np.load(Path(__file__).with_name("controller_data.npz"), allow_pickle=False) as data:
        return data[f"{name}_K"] @ data[f"{name}_E"]
