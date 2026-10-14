"""Data loading, synthetic data and train/val/test splitting."""
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def make_synthetic(n_per_class=120, size=32, n_classes=4, n_groups=12, looks=4, seed=0):
    """Synthetic SAR patches for smoke tests only."""
    rng = np.random.default_rng(seed)
    vv_db = np.linspace(-20, -6, n_classes)
    vh_db = vv_db - np.linspace(7, 11, n_classes)
    X, y, groups = [], [], []
    for c in range(n_classes):
        for i in range(n_per_class):
            g = int(rng.integers(0, n_groups))
            shift = rng.normal(0, 1.0)
            base = np.stack([np.full((size, size), vv_db[c] + shift),
                             np.full((size, size), vh_db[c] + shift)])
            lin = 10 ** (base / 10)
            speckle = rng.gamma(looks, 1.0 / looks, size=lin.shape)
            X.append(lin * speckle)
            y.append(c)
            groups.append(f"scene{g}")
    names = [f"class_{c}" for c in range(n_classes)]
    return np.asarray(X, dtype=np.float32), np.asarray(y), np.asarray(groups), names


def _read_patch(path):
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix in (".tif", ".tiff"):
        import rasterio  # optional dependency
        with rasterio.open(path) as src:
            return src.read([1, 2])
    return None


def load_folder(root, input_db=False, max_per_class=0, seed=0):
    """Read <root>/<Class>/<scene>__<idx>.(npy|tif) into (X, y, groups, class_names)."""
    root = Path(root)
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not classes:
        raise FileNotFoundError(f"No class folders found in {root}")
    rng = np.random.default_rng(seed)
    X, y, groups = [], [], []
    for ci, cname in enumerate(classes):
        files = sorted(f for f in (root / cname).iterdir() if f.suffix in (".npy", ".tif", ".tiff"))
        if max_per_class and len(files) > max_per_class:
            files = list(rng.choice(np.array(files, dtype=object), max_per_class, replace=False))
        for f in files:
            arr = _read_patch(f)
            if arr is None or arr.ndim != 3 or arr.shape[0] != 2:
                warnings.warn(f"Skipping {f}: expected shape (2, H, W)")
                continue
            arr = arr.astype(np.float32)
            if input_db:
                arr = 10 ** (arr / 10)
            X.append(arr)
            y.append(ci)
            groups.append(f.stem.split("__")[0] if "__" in f.stem else f"{cname}/{f.stem}")
    if not X:
        raise FileNotFoundError(f"No valid (2, H, W) patches found under {root}")
    shapes = {x.shape for x in X}
    if len(shapes) != 1:
        raise ValueError(f"All patches must have the same shape, found: {sorted(shapes)[:5]}")
    return np.stack(X), np.asarray(y), np.asarray(groups), classes


def split_indices(y, groups, mode="group", seed=0):
    """70/15/15 split. mode="group" keeps each scene in one part."""
    idx = np.arange(len(y))
    if mode == "random":
        tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
        va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=seed)
        return tr, va, te
    if mode != "group":
        raise ValueError("mode must be 'group' or 'random'")
    if len(set(groups)) == len(groups):
        warnings.warn("Every patch is its own group, so the group split is effectively random. "
                      "Name files <scene>__<idx> for a scene-level split.")
    classes = set(y.tolist())
    for attempt in range(200):
        rs = seed + attempt
        trva, te = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=rs).split(idx, y, groups))
        tr_rel, va_rel = next(GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=rs)
                              .split(trva, y[trva], groups[trva]))
        tr, va = trva[tr_rel], trva[va_rel]
        if all(set(y[p].tolist()) == classes for p in (tr, va, te)):
            return tr, va, te
    warnings.warn("No group split found with every class in train, val and test; check class counts.")
    return tr, va, te
