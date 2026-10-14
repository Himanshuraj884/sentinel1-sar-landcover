import numpy as np

from src.data import make_synthetic, split_indices
from src.sar import FEATURE_NAMES, extract_features, lee_filter, preprocess, to_cnn_input, to_db


def test_to_db_known_values():
    assert np.isclose(to_db(np.array([1.0]))[0], 0.0)
    assert np.isclose(to_db(np.array([0.1]))[0], -10.0)


def test_lee_filter_reduces_variance_and_keeps_shape():
    rng = np.random.default_rng(0)
    img = rng.gamma(4, 0.25, size=(32, 32)).astype(np.float32)
    out = lee_filter(img, 5)
    assert out.shape == img.shape and out.var() < img.var()


def test_features_shape_and_finite():
    X, y, g, names = make_synthetic(n_per_class=10, size=16)
    F = extract_features(preprocess(X))
    assert F.shape == (len(y), len(FEATURE_NAMES)) and np.isfinite(F).all()


def test_cnn_input_range():
    X, *_ = make_synthetic(n_per_class=5, size=16)
    C = to_cnn_input(preprocess(X))
    assert C.shape[-1] == 3 and C.min() >= 0 and C.max() <= 1


def test_group_split_has_no_group_overlap():
    X, y, g, _ = make_synthetic(n_per_class=60, size=8, n_groups=10)
    tr, va, te = split_indices(y, g, "group", seed=1)
    s_tr, s_va, s_te = set(g[tr]), set(g[va]), set(g[te])
    assert not (s_tr & s_te) and not (s_tr & s_va) and not (s_va & s_te)
    assert len(set(tr) | set(va) | set(te)) == len(tr) + len(va) + len(te)


def test_random_split_covers_all_classes():
    X, y, g, _ = make_synthetic(n_per_class=40, size=8)
    tr, va, te = split_indices(y, g, "random", seed=0)
    assert set(y[te]) == set(y)


def test_group_split_keeps_every_class_in_every_part():
    X, y, g, _ = make_synthetic(n_per_class=60, size=8, n_groups=6, seed=3)
    for seed in range(5):
        tr, va, te = split_indices(y, g, "group", seed)
        for part in (tr, va, te):
            assert set(y[part]) == set(y)


def test_rf_runs_and_beats_chance(tmp_path):
    from src.sar import extract_features, preprocess
    from src.train_rf import run_rf
    X, y, g, names = make_synthetic(n_per_class=60, size=16)
    F = extract_features(preprocess(X))
    tr, va, te = split_indices(y, g, "group", 0)
    res = run_rf(F, y, tr, va, te, g, names, tmp_path, "group")
    assert res["test_accuracy"] > 1.5 / len(names)


def test_rf_does_not_crash_when_a_class_is_missing_from_test(tmp_path):
    from src.sar import extract_features, preprocess
    from src.train_rf import run_rf
    X, y, g, names = make_synthetic(n_per_class=40, size=16)
    F = extract_features(preprocess(X))
    tr, va, te = split_indices(y, g, "random", 0)
    te = te[y[te] != 0]
    run_rf(F, y, tr, va, te, g, names, tmp_path, "random")


def test_end_to_end_cli(tmp_path, monkeypatch):
    import json, sys
    import run_all
    monkeypatch.setattr(sys, "argv", ["run_all.py", "--synthetic", "--skip-cnn", "--out", str(tmp_path)])
    run_all.main()
    s = json.loads((tmp_path / "reports" / "summary.json").read_text())
    assert s["synthetic"] and s["random_forest"]["test_accuracy"] > 0.5
    assert (tmp_path / "figures" / "03_rf_confusion.png").exists()


def test_loader_roundtrip_and_scene_grouping(tmp_path):
    from src.data import load_folder
    rng = np.random.default_rng(0)
    for cls in ("A", "B"):
        (tmp_path / cls).mkdir()
        for i in range(3):
            np.save(tmp_path / cls / f"sceneX__{i}.npy", rng.random((2, 8, 8)).astype(np.float32))
    X, y, g, names = load_folder(tmp_path)
    assert X.shape == (6, 2, 8, 8) and names == ["A", "B"]
    assert set(g) == {"sceneX"}
