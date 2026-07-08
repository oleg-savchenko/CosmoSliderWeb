"""Verification layer 1: does our hand-derived NumPy forward pass match the real
cosmopower_jax package's own predictions? Pure algorithm check, no TF/Keras involved.

Runs with either env (only needs numpy). Use cpj:
    /Users/olegsavchenko/miniforge3/envs/cpj/bin/python verify_layer1_numpy_oracle.py
"""
import numpy as np
from reference_forward_numpy import load_npz_model, forward, forward_nonlinear

vec = np.load("verification_data/vectors.npz")
model_lin = load_npz_model("artifacts/mpk_lin.npz")
model_boost = load_npz_model("artifacts/mpk_boost.npz")


def max_rel_err(pred, truth):
    return np.max(np.abs(pred - truth) / np.abs(truth))


pred_lin = forward(model_lin, vec["x_lin"])
err_lin = max_rel_err(pred_lin, vec["y_lin"])
print(f"mpk_lin    max relative error vs CosmoPowerJAX: {err_lin:.3e}")

pred_boost = forward(model_boost, vec["x_boost"])
err_boost = max_rel_err(pred_boost, vec["y_boost"])
print(f"mpk_boost  max relative error vs CosmoPowerJAX: {err_boost:.3e}")

pred_nonlin = forward_nonlinear(model_boost, model_lin, vec["x_nonlin"])
err_nonlin = max_rel_err(pred_nonlin, vec["y_nonlin"])
print(f"mpk_nonlin max relative error vs CosmoPowerJAX: {err_nonlin:.3e}")

TOL = 1e-4
assert err_lin < TOL, f"mpk_lin error {err_lin} exceeds tolerance {TOL}"
assert err_boost < TOL, f"mpk_boost error {err_boost} exceeds tolerance {TOL}"
assert err_nonlin < TOL, f"mpk_nonlin error {err_nonlin} exceeds tolerance {TOL}"
print(f"\nLayer 1 PASSED (all errors < {TOL:.0e})")
