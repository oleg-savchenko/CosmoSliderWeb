"""Sample random in-domain parameter vectors, run the REAL cosmopower_jax models,
and save (inputs, outputs) as test vectors for downstream verification layers.

Run with the `cpj` conda env:
    /Users/olegsavchenko/miniforge3/envs/cpj/bin/python generate_verification_vectors.py
"""
import json
import numpy as np
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX

N_SAMPLES = 200
SAFETY_MARGIN = 0.9
OUT_DIR = "verification_data"

rng = np.random.RandomState(42)


def sample_uniform_box(mean, std, n):
    """Recover a safety-margined uniform box from stored training mean/std.

    For X ~ Uniform(a, b): mean=(a+b)/2, std=(b-a)/(2*sqrt(3)).
    So half-width = sqrt(3)*std; shrink by SAFETY_MARGIN to stay off domain edges.
    """
    half_width = np.sqrt(3) * std * SAFETY_MARGIN
    lo, hi = mean - half_width, mean + half_width
    return rng.uniform(lo, hi, size=(n, len(mean)))


def run_probe(probe, n_params_expected):
    emu = CosmoPowerJAX(probe=probe)
    mean, std = np.asarray(emu.param_train_mean), np.asarray(emu.param_train_std)
    assert len(mean) == n_params_expected, f"{probe}: expected {n_params_expected} params, got {len(mean)}"
    x = sample_uniform_box(mean, std, N_SAMPLES)
    y = np.asarray(emu.predict(x))
    return x, y


x_lin, y_lin = run_probe("mpk_lin", 6)
x_boost, y_boost = run_probe("mpk_boost", 8)
x_nonlin, y_nonlin = run_probe("mpk_nonlin", 8)

np.savez(f"{OUT_DIR}/vectors.npz",
         x_lin=x_lin, y_lin=y_lin,
         x_boost=x_boost, y_boost=y_boost,
         x_nonlin=x_nonlin, y_nonlin=y_nonlin)

# also dump JSON for the Node.js verification layer
with open(f"{OUT_DIR}/vectors.json", "w") as f:
    json.dump({
        "x_lin": x_lin.tolist(), "y_lin": y_lin.tolist(),
        "x_nonlin": x_nonlin.tolist(), "y_nonlin": y_nonlin.tolist(),
    }, f)

print(f"lin:     x{x_lin.shape} -> y{y_lin.shape}, y range [{y_lin.min():.4g}, {y_lin.max():.4g}]")
print(f"boost:   x{x_boost.shape} -> y{y_boost.shape}, y range [{y_boost.min():.4g}, {y_boost.max():.4g}]")
print(f"nonlin:  x{x_nonlin.shape} -> y{y_nonlin.shape}, y range [{y_nonlin.min():.4g}, {y_nonlin.max():.4g}]")
