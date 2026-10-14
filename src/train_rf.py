"""Random Forest, cross-validation and polarisation ablation."""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score)
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score

from .sar import FEATURE_NAMES


def _rf(seed):
    return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed, class_weight="balanced")


def _cols(prefixes):
    return [i for i, n in enumerate(FEATURE_NAMES) if any(n.startswith(p) for p in prefixes)]


def run_rf(F, y, tr, va, te, groups, class_names, out_dir, split_mode="group", seed=0):
    out_dir = Path(out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    model = _rf(seed).fit(F[tr], y[tr])
    pred = model.predict(F[te])
    labels = list(range(len(class_names)))
    present = sorted(set(y[te].tolist()))
    res = {
        "val_accuracy": float(accuracy_score(y[va], model.predict(F[va]))),
        "test_accuracy": float(accuracy_score(y[te], pred)),
        "test_macro_f1": float(f1_score(y[te], pred, labels=present, average="macro", zero_division=0)),
        "report": classification_report(y[te], pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0),
    }

    trva = np.concatenate([tr, va])
    n_groups = len(set(groups[trva]))
    if split_mode == "group" and n_groups >= 5:
        cv = GroupKFold(n_splits=5)
        scores = cross_val_score(_rf(seed), F[trva], y[trva], cv=cv, groups=groups[trva])
        res["cv_kind"] = "GroupKFold(5)"
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        scores = cross_val_score(_rf(seed), F[trva], y[trva], cv=cv)
        res["cv_kind"] = "StratifiedKFold(5)"
    res["cv_accuracy_mean"], res["cv_accuracy_std"] = float(scores.mean()), float(scores.std())

    ablation = {}
    for label, prefixes in {"VV only": ["vv_"], "VH only": ["vh_"],
                            "VV + VH": ["vv_", "vh_"], "all features": [""]}.items():
        c = _cols(prefixes)
        m = _rf(seed).fit(F[tr][:, c], y[tr])
        ablation[label] = float(accuracy_score(y[te], m.predict(F[te][:, c])))
    res["ablation_test_accuracy"] = ablation

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y[te], pred, labels=labels, display_labels=class_names, ax=ax,
                                            xticks_rotation=45, colorbar=False)
    fig.tight_layout(); fig.savefig(out_dir / "figures" / "03_rf_confusion.png", dpi=130); plt.close(fig)

    order = np.argsort(model.feature_importances_)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh([FEATURE_NAMES[i] for i in order], model.feature_importances_[order])
    ax.set_title("Random Forest feature importance")
    fig.tight_layout(); fig.savefig(out_dir / "figures" / "04_rf_feature_importance.png", dpi=130); plt.close(fig)

    joblib.dump(model, out_dir / "models" / "random_forest.joblib")
    return res
