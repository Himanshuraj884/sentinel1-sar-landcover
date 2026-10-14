"""Speckle filtering, dB conversion and per-patch features."""
import numpy as np
from scipy.ndimage import uniform_filter

EPS = 1e-6
VV, VH = 0, 1


def to_db(x):
    return 10.0 * np.log10(np.clip(x, EPS, None))


def lee_filter(img, size=5):
    """Lee filter on one linear-intensity image."""
    mean = uniform_filter(img, size)
    sq_mean = uniform_filter(img * img, size)
    var = np.clip(sq_mean - mean * mean, 0, None)
    overall = np.var(img)
    w = var / (var + overall + EPS)
    return mean + w * (img - mean)


def preprocess(X_lin, despeckle=True, size=5):
    """Linear (N, 2, H, W) to dB, optionally Lee-filtered first."""
    X = X_lin.astype(np.float32)
    if despeckle:
        out = np.empty_like(X)
        for i in range(X.shape[0]):
            for c in range(2):
                out[i, c] = lee_filter(X[i, c], size)
        X = out
    return to_db(X)


FEATURE_NAMES = (
    [f"vv_{s}" for s in ("mean", "std", "p10", "p90")]
    + [f"vh_{s}" for s in ("mean", "std", "p10", "p90")]
    + ["ratio_mean", "ratio_std", "rvi_mean", "vv_texture", "vh_texture"]
)


def extract_features(X_db):
    """13 features per patch from dB data (N, 2, H, W)."""
    feats = []
    for p in X_db:
        vv, vh = p[VV], p[VH]
        ratio = vv - vh
        lin_vv, lin_vh = 10 ** (vv / 10), 10 ** (vh / 10)
        rvi = 4 * lin_vh / (lin_vv + lin_vh + EPS)
        tex = lambda a: np.mean(np.abs(np.diff(a, axis=0))) + np.mean(np.abs(np.diff(a, axis=1)))
        row = []
        for a in (vv, vh):
            row += [a.mean(), a.std(), np.percentile(a, 10), np.percentile(a, 90)]
        row += [ratio.mean(), ratio.std(), rvi.mean(), tex(vv), tex(vh)]
        feats.append(row)
    return np.asarray(feats, dtype=np.float32)


def to_cnn_input(X_db):
    """Scale VV, VH (-30 to 5 dB) and VV-VH (0 to 20 dB) to [0, 1]; returns (N, H, W, 3)."""
    vv = np.clip((X_db[:, VV] + 30) / 35, 0, 1)
    vh = np.clip((X_db[:, VH] + 30) / 35, 0, 1)
    ratio = np.clip((X_db[:, VV] - X_db[:, VH]) / 20, 0, 1)
    return np.stack([vv, vh, ratio], axis=-1).astype(np.float32)
