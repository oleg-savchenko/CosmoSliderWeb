"""Verification layer 2: does the reloaded tf.saved_model match the NumPy oracle
and the real CosmoPowerJAX-generated test vectors?

Run with the `mpk2tfjs` conda env:
    /Users/olegsavchenko/miniforge3/envs/mpk2tfjs/bin/python verify_layer2_savedmodel.py
"""
import numpy as np
import tensorflow as tf
from reference_forward_numpy import load_npz_model, forward, forward_nonlinear

vec = np.load("verification_data/vectors.npz")
model_lin_np = load_npz_model("artifacts/mpk_lin.npz")
model_boost_np = load_npz_model("artifacts/mpk_boost.npz")

TOL = 1e-3


def max_rel_err(pred, truth):
    return np.max(np.abs(pred - truth) / np.abs(truth))


def run_saved_model(path, x):
    loaded = tf.saved_model.load(path)
    infer = loaded.signatures["serving_default"]
    x_tf = tf.constant(x, dtype=tf.float32)
    out = infer(x_tf)
    # single-output signature -> grab the only value
    return list(out.values())[0].numpy()


# --- mpk_lin ---
pred_lin_np = forward(model_lin_np, vec["x_lin"], dtype=np.float32)
pred_lin_sm = run_saved_model("artifacts/saved_model_lin", vec["x_lin"].astype(np.float32))
err_lin_vs_np = max_rel_err(pred_lin_sm, pred_lin_np)
err_lin_vs_jax = max_rel_err(pred_lin_sm, vec["y_lin"])
print(f"mpk_lin    SavedModel vs NumPy oracle: {err_lin_vs_np:.3e}   vs JAX: {err_lin_vs_jax:.3e}")

# --- mpk_nonlin ---
pred_nonlin_np = forward_nonlinear(model_boost_np, model_lin_np, vec["x_nonlin"], dtype=np.float32)
pred_nonlin_sm = run_saved_model("artifacts/saved_model_nonlin", vec["x_nonlin"].astype(np.float32))
err_nonlin_vs_np = max_rel_err(pred_nonlin_sm, pred_nonlin_np)
err_nonlin_vs_jax = max_rel_err(pred_nonlin_sm, vec["y_nonlin"])
print(f"mpk_nonlin SavedModel vs NumPy oracle: {err_nonlin_vs_np:.3e}   vs JAX: {err_nonlin_vs_jax:.3e}")

assert err_lin_vs_np < TOL, f"mpk_lin vs NumPy oracle error {err_lin_vs_np} exceeds {TOL}"
assert err_lin_vs_jax < TOL, f"mpk_lin vs JAX error {err_lin_vs_jax} exceeds {TOL}"
assert err_nonlin_vs_np < TOL, f"mpk_nonlin vs NumPy oracle error {err_nonlin_vs_np} exceeds {TOL}"
assert err_nonlin_vs_jax < TOL, f"mpk_nonlin vs JAX error {err_nonlin_vs_jax} exceeds {TOL}"
print(f"\nLayer 2 PASSED (all errors < {TOL:.0e})")
