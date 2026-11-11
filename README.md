# Flexible-arm data for competition

This is a small port of the flexible arm and the two controllers in the motivating
example of *Learning Stabilizing Latent Feedback Controllers via Multi-Step
Backward Losses* (https://arxiv.org/abs/2610.40931). It generates full-state trajectories of the 40-dimensional arm
under either controller. The trained feedback matrices are included.

Use Python 3.10 or later:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py --controller both --trajectories 16 --steps 200 --seed 0
```

This writes `data/stable.npz` and `data/unstable.npz`, each covering up to 10 seconds
at a sampling period of 0.05 s. Both runs start from the same initial states,
drawn independently from a zero-mean Gaussian with standard deviation 0.01 for
each joint angle and angular velocity. Use `--initial-scale` to change that
standard deviation, `--seed` for another draw, and `--output` for another directory.
An existing output file with the same name is overwritten.

The default plant is non-linear. For its linearization about the upright
equilibrium, run:

```bash
python generate_data.py --plant linear --controller both --output data/linear
```

`stable` uses the model learned with the multi-step backward loss, and `unstable`
uses the model learned with the decoder-free forward loss. Both apply the linear
feedback law from the motivating example, `u = K J_E(0) x`, with physical torque
clipped to ±40 N m and held between samples. Recomputing the unsaturated
closed-loop spectral radii from the included matrices gives approximately
0.9549 and 1.1294, respectively. These describe local stability at the upright
equilibrium. The stable controller is not guaranteed to recover from arbitrary
initial states. There is no process or observation noise.

## Reading the data

```python
import numpy as np

with np.load("data/unstable.npz", allow_pickle=False) as data:
    i = 0
    transitions = int(data["lengths"][i])
    states = data["X"][i, :transitions + 1]
    torque = data["torque"][i, :transitions]
    time = data["time"][:transitions + 1]
    print(data["status"][i])
```

For `N` trajectories and `T` steps, the files contain:

| Key | Shape | Meaning |
| --- | --- | --- |
| `X` | `(N, T+1, 40)` | First 20 entries: relative joint angles in rad; last 20: joint rates in rad/s |
| `U` | `(N, T, 1)` | Applied input in normalized units |
| `torque` | `(N, T, 1)` | Same applied torque at every joint, in N m |
| `time` | `(T+1,)` | Sample times in seconds |
| `lengths` | `(N,)` | Number of stored transitions per trajectory |
| `status` | `(N,)` | `completed`, `fallen`, or `numerical_failure` |

Scalar metadata records `controller`, `plant`, `dt`, `u_scale`, `torque_limit`,
`seed`, and `initial_scale`. The normalization is `U = u_scale * torque`, with
`u_scale` approximately 2.13635, so the discrete input matrix has Euclidean norm 1.

For the non-linear plant, a trajectory stops at the first sampled state where
any section is more than 90 degrees from upright.
the model has no ground-contact dynamics. An initial state already beyond this
angle has length zero. The linear plant has no angle cutoff. A numerical
failure retains only the preceding valid states. Entries after each valid
prefix are `NaN`, including unused inputs. Never treat that padding as data.
`completed` means the requested observation window ended -- it does not establish
stability. The command returns a nonzero exit status if a numerical failure occurs.

## Python interface

```python
import numpy as np
from flexible_arm import FlexibleArm, generate_data, load_controller

arm = FlexibleArm()
initial_states = 0.01 * np.random.default_rng(7).standard_normal((8, 40))
data = generate_data("stable", initial_states=initial_states, steps=200, arm=arm)

# For a custom simulation, load the unsaturated normalized feedback gain.
gain = load_controller("stable")                 # shape (1, 40)
x = initial_states[0]
u = np.clip(gain @ x, -40 * arm.u_scale, 40 * arm.u_scale)
next_x = arm.step(x, u)                           # non-linear; one 0.05 s period
next_x_linear = arm.step(x, u, plant="linear")
```

`arm.A` and `arm.B` give the normalized discrete linearization, and
`arm.centerline(q)` returns the section endpoint coordinates in meters.
`arm.step` also accepts batches of states `(N, 40)` and inputs `(N, 1)`.
It does not clip the input or stop on a fall. `generate_data` applies those rules.

## Source and verification

The source repository is
`~/Documents/Learning-Stabilizing-Latent-Feedback-Controllers`, at commit
`92231e2abbf0ed7e127d17d5b8c07c1e771d4bea`. The port retains the default arm from
`latentctl/systems.py`, including its configuration-dependent inertia and
momentum integrator with 50 substeps per sample. The two feedback gains come
from these `results/paper/motivating_example/` checkpoints at horizon 40:

- `mlp__nonlinear__s0__sr__hT.pkl` → `stable`
- `mlp__nonlinear__s0__Lf__sr__hT.pkl` → `unstable`

`flexible_arm/controller_data.npz` stores each encoder Jacobian `E`, latent
linearization `L, H`, and latent LQR gain `K`, along with the source revision,
checkpoint names, and checkpoint SHA-256 hashes. The LQR uses `Q = I_4`, `R = I_1`
and the sign convention `u = K z`. Spectral radii above are recomputed from these
parameters rather than taken from cached experiment diagnostics.
To reproduce the export from the trusted local checkpoints:

```bash
python tools/export_controllers.py
```

Run the checks with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

The checks include the local linearization, comparison of a non-linear step
against an independent Radau integration of the mechanical equations,
closed-loop behavior, torque units and saturation, and saved-data boundaries.
The port was also checked directly against the source simulator.

`system_description.tex` is a standalone description of the mechanics,
controllers, and implementation.
To rebuild it:

```bash
pdflatex -halt-on-error system_description.tex
pdflatex -halt-on-error system_description.tex
```
