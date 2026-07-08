"""Framework-agnostic NumPy re-implementation of the cosmopower_jax forward pass.

Depends on nothing but NumPy - importable from either the `cpj` or `mpk2tfjs` env.
Used as an independent oracle to separate "is the algorithm understanding correct"
from "is the Keras/TF.js port correct".
"""
import numpy as np


def load_npz_model(path):
    d = np.load(path)
    n_hidden = sum(1 for k in d.files if k.startswith("alpha"))
    return dict(
        weights=[(d[f"W{i}"], d[f"b{i}"]) for i in range(n_hidden + 1)],
        hyper=[(d[f"alpha{i}"], d[f"beta{i}"]) for i in range(n_hidden)],
        param_mean=d["param_train_mean"],
        param_std=d["param_train_std"],
        feat_mean=d["feature_train_mean"],
        feat_std=d["feature_train_std"],
    )


def forward(model, x, dtype=np.float64):
    """x: shape (n_samples, n_params) in physical units. Returns P(k), shape (n_samples, 420)."""
    z = (np.asarray(x, dtype) - model["param_mean"].astype(dtype)) / model["param_std"].astype(dtype)
    for (W, b), (a, beta) in zip(model["weights"][:-1], model["hyper"]):
        W, b, a, beta = (v.astype(dtype) for v in (W, b, a, beta))
        pre = z @ W.T + b
        sig = 1.0 / (1.0 + np.exp(-a * pre))
        z = (beta + sig * (1.0 - beta)) * pre

    W, b = (v.astype(dtype) for v in model["weights"][-1])
    preds = z @ W.T + b[-1]  # scalar-bias quirk: only the last element of b is real
    preds = preds * model["feat_std"].astype(dtype) + model["feat_mean"].astype(dtype)
    return 10.0 ** preds


def forward_nonlinear(model_boost, model_lin, x8, dtype=np.float64):
    """x8: shape (n_samples, 8) = omega_b, omega_cdm, h, n_s, ln10^{10}A_s, cmin, eta_0, z."""
    boost = forward(model_boost, x8, dtype=dtype)
    x6 = np.concatenate([x8[:, :5], x8[:, 7:8]], axis=1)
    lin = forward(model_lin, x6, dtype=dtype)
    return boost * lin
