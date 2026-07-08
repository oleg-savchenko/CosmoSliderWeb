"""Rebuild mpk_lin and mpk_nonlin as tf.keras models from the extracted .npz weights,
then export as SavedModel (the same pipeline that produced the existing CMB model).

Run with the `mpk2tfjs` conda env (TF 2.15.1 / Keras 2.15, pinned to stay off Keras 3):
    /Users/olegsavchenko/miniforge3/envs/mpk2tfjs/bin/python build_keras_models.py
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

ARTIFACTS = "artifacts"


def build_subnetwork(x, npz_path, prefix):
    """x: a KerasTensor with the right input width for this sub-network. Returns P(k), shape (None, 420)."""
    d = np.load(npz_path)
    n_hidden = sum(1 for k in d.files if k.startswith("alpha"))
    mean = d["param_train_mean"].astype("float32")
    std = d["param_train_std"].astype("float32")
    fmean = d["feature_train_mean"].astype("float32")
    fstd = d["feature_train_std"].astype("float32")

    z = layers.Normalization(axis=-1, mean=mean, variance=std ** 2,
                              name=f"{prefix}_normalization")(x)

    for i in range(n_hidden):
        W = d[f"W{i}"].astype("float32")
        b = d[f"b{i}"].astype("float32")
        dense = layers.Dense(W.shape[0], activation=None, name=f"{prefix}_dense_{i}")
        dense.build(z.shape)
        dense.set_weights([W.T, b])
        pre = dense(z)

        a_t = tf.constant(d[f"alpha{i}"].astype("float32"))
        beta_t = tf.constant(d[f"beta{i}"].astype("float32"))
        z = layers.Lambda(
            lambda t, a=a_t, be=beta_t: (be + tf.sigmoid(a * t) * (1.0 - be)) * t,
            name=f"{prefix}_activation_{i}",
        )(pre)

    Wf = d[f"W{n_hidden}"].astype("float32")
    bf = d[f"b{n_hidden}"].astype("float32")
    scalar_bias = np.full(Wf.shape[0], bf[-1], dtype="float32")
    out_dense = layers.Dense(Wf.shape[0], activation=None, name=f"{prefix}_output_dense")
    out_dense.build(z.shape)
    out_dense.set_weights([Wf.T, scalar_bias])
    logpk = out_dense(z)

    return layers.Lambda(
        lambda t: tf.pow(10.0, t * fstd + fmean),
        name=f"{prefix}_unnormalize_pow10",
    )(logpk)


# --- (a) mpk_lin standalone: 6 -> 420 ---
inp_lin = Input(shape=(6,), name="input_1")
out_lin = build_subnetwork(inp_lin, f"{ARTIFACTS}/mpk_lin.npz", "lin")
model_lin = Model(inp_lin, out_lin, name="mpk_lin")
tf.saved_model.save(model_lin, f"{ARTIFACTS}/saved_model_lin")
print(f"mpk_lin SavedModel written, output shape {model_lin.output_shape}")

# --- (b) mpk_nonlin combined: 8 -> 420, boost(x) * lin(slice(x)) ---
inp_nonlin = Input(shape=(8,), name="input_1")
pk_boost = build_subnetwork(inp_nonlin, f"{ARTIFACTS}/mpk_boost.npz", "boost")
lin_input = layers.Lambda(
    lambda t: tf.concat([t[:, 0:5], t[:, 7:8]], axis=1),
    name="slice_lin_input",
)(inp_nonlin)
pk_lin = build_subnetwork(lin_input, f"{ARTIFACTS}/mpk_lin.npz", "lin2")
out_nonlin = layers.Multiply(name="boost_times_lin")([pk_boost, pk_lin])
model_nonlin = Model(inp_nonlin, out_nonlin, name="mpk_nonlin")
tf.saved_model.save(model_nonlin, f"{ARTIFACTS}/saved_model_nonlin")
print(f"mpk_nonlin SavedModel written, output shape {model_nonlin.output_shape}")
