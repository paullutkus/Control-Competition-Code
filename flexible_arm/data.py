"""Closed-loop trajectories with full states and applied inputs."""

import numpy as np

from .controllers import load_controller
from .system import FlexibleArm


def generate_data(controller="stable", *, plant="nonlinear", trajectories=16,
                  steps=200, seed=0, initial_scale=0.01, initial_states=None,
                  arm=None):
    """Generate deterministic trajectories; randomness is only in the initial states.

    With initial_states=None, draw 'trajectories' states from N(0, initial_scale**2 I).
    Otherwise use the supplied (N, 40) array, ignoring trajectories and initial_scale.
    The same seed gives the same initial states under either controller or plant.

    X has shape (N, steps+1, 40); U and torque have shape (N, steps, 1).
    U is the normalized input and torque = U / u_scale is in N m. Feedback is
    clipped to +/-40 N m, then held for dt=0.05 s. There is no added noise.

    A nonlinear rollout stops when any section exceeds 90 degrees from upright.
    The first fallen state is retained. A linear rollout has no angle cutoff.
    A non-finite step or failed linear solve stops only the affected trajectory.
    lengths[i] counts stored transitions: X[i, :lengths[i]+1] and
    U[i, :lengths[i]] are valid. Unused entries are NaN. status is 'completed',
    'fallen', or 'numerical_failure'; it is not a stability classification.
    """
    if plant not in ("linear", "nonlinear"):
        raise ValueError("plant must be 'linear' or 'nonlinear'")
    if not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    arm = FlexibleArm() if arm is None else arm
    gain = load_controller(controller)
    if initial_states is None:
        if not isinstance(trajectories, (int, np.integer)) or trajectories < 1:
            raise ValueError("trajectories must be a positive integer")
        if not np.isfinite(initial_scale) or initial_scale < 0:
            raise ValueError("initial_scale must be finite and nonnegative")
        x0 = initial_scale * np.random.default_rng(seed).standard_normal((trajectories, arm.n))
    else:
        x0 = np.atleast_2d(np.asarray(initial_states, dtype=float))
    if x0.ndim != 2 or x0.shape[1] != arm.n or len(x0) == 0 or not np.all(np.isfinite(x0)):
        raise ValueError("initial_states must be a finite, nonempty array of shape (N, 40)")

    count = len(x0)
    X = np.full((count, steps + 1, arm.n), np.nan)
    U = np.full((count, steps, arm.p), np.nan)
    X[:, 0] = x0
    lengths = np.zeros(count, dtype=int)
    status = np.full(count, "completed", dtype="U17")
    active = np.ones(count, dtype=bool)
    if plant == "nonlinear":
        active = ~arm.fallen(x0)
        status[~active] = "fallen"

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for t in range(steps):
            ids = np.flatnonzero(active)
            if not len(ids):
                break
            states = X[ids, t]
            inputs = np.clip(states @ gain.T, -arm.u_max * arm.u_scale, arm.u_max * arm.u_scale)
            try:
                next_states = arm.step(states, inputs, plant)
            except np.linalg.LinAlgError:
                # A singular solve in one member must not discard the other members.
                next_states = np.full_like(states, np.nan)
                for j in range(len(ids)):
                    try:
                        next_states[j] = arm.step(states[j], inputs[j], plant)
                    except np.linalg.LinAlgError:
                        pass
            finite = np.all(np.isfinite(next_states), axis=1)
            bad = ids[~finite]
            status[bad], active[bad] = "numerical_failure", False
            good = ids[finite]
            X[good, t + 1] = next_states[finite]
            U[good, t] = inputs[finite]
            lengths[good] = t + 1
            if plant == "nonlinear":
                fallen = good[arm.fallen(next_states[finite])]
                status[fallen], active[fallen] = "fallen", False

    return dict(X=X, U=U, torque=U / arm.u_scale,
                time=np.arange(steps + 1) * arm.dt, lengths=lengths, status=status,
                controller=controller, plant=plant, dt=arm.dt,
                u_scale=arm.u_scale, torque_limit=arm.u_max)
