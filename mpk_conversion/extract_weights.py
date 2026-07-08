"""Extract cosmopower_jax mpk_lin/mpk_boost weights into plain-NumPy .npz files.

Run with the `cpj` conda env (has jax + cosmopower_jax installed):
    /Users/olegsavchenko/miniforge3/envs/cpj/bin/python extract_weights.py
"""
import numpy as np
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX

OUT_DIR = "artifacts"


def extract(probe, out_name):
    emu = CosmoPowerJAX(probe=probe)

    weights = [np.asarray(w, dtype=np.float64) for w, b in emu.weights]
    biases = [np.asarray(b, dtype=np.float64) for w, b in emu.weights]
    alphas = [np.asarray(a, dtype=np.float64) for a, b in emu.hyper_params]
    betas = [np.asarray(b, dtype=np.float64) for a, b in emu.hyper_params]

    final_b = biases[-1]
    nz = np.flatnonzero(final_b)
    assert len(nz) <= 1 and (len(nz) == 0 or nz[0] == len(final_b) - 1), (
        f"{probe}: scalar-bias assumption violated (nonzero final-bias indices: {nz}), "
        "re-check before exporting!"
    )

    np.savez(
        f"{OUT_DIR}/{out_name}.npz",
        **{f"W{i}": w for i, w in enumerate(weights)},
        **{f"b{i}": b for i, b in enumerate(biases)},
        **{f"alpha{i}": a for i, a in enumerate(alphas)},
        **{f"beta{i}": b for i, b in enumerate(betas)},
        param_train_mean=np.asarray(emu.param_train_mean, dtype=np.float64),
        param_train_std=np.asarray(emu.param_train_std, dtype=np.float64),
        feature_train_mean=np.asarray(emu.feature_train_mean, dtype=np.float64),
        feature_train_std=np.asarray(emu.feature_train_std, dtype=np.float64),
        parameters=np.asarray(emu.parameters),
    )
    print(f"{probe}: extracted {len(weights)} layers, "
          f"architecture in->out sizes = {[w.shape for w in weights]}, "
          f"params = {list(emu.parameters)}")
    return emu


emu_lin = extract("mpk_lin", "mpk_lin")
emu_boost = extract("mpk_boost", "mpk_boost")

# k-grid sidecar: identical between the two models, verify that before trusting either copy
assert np.allclose(np.asarray(emu_lin.modes), np.asarray(emu_boost.modes)), \
    "mpk_lin and mpk_boost k-grids differ - investigate before proceeding!"
np.save(f"{OUT_DIR}/modes.npy", np.asarray(emu_lin.modes, dtype=np.float64))
print(f"modes: {emu_lin.modes.shape[0]} k-values, range [{emu_lin.modes[0]:.6g}, {emu_lin.modes[-1]:.6g}]")
